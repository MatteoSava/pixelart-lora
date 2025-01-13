from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from train_jalb.config import TrainingConfig
from train_jalb.data import PixelArtDataset, collate
from train_jalb.quantization import SixGrayQuantizer
from train_jalb.sdxl import encode_prompts, make_time_ids, predicted_x0

logger = logging.getLogger(__name__)


def _set_hf_cache(cache_dir: str | None) -> None:
    if cache_dir is None:
        return
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(cache_dir) / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_dir) / "hub"))


def _trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def inject_lora(unet: UNet2DConditionModel, rank: int = 4) -> list[torch.nn.Parameter]:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.requires_grad_(False)
    unet.add_adapter(lora_config)
    return _trainable_parameters(unet)


def save_lora_weights(unet: UNet2DConditionModel, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
    )


def train(config: TrainingConfig) -> str:
    config.validate()
    _set_hf_cache(config.hf_cache_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(config.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(config.pretrained_model, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model, subfolder="text_encoder").eval()
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        config.pretrained_model,
        subfolder="text_encoder_2",
    ).eval()
    vae = AutoencoderKL.from_pretrained(config.pretrained_model, subfolder="vae").eval()
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    trainable = inject_lora(unet, rank=config.lora_rank)

    quantizer = SixGrayQuantizer()
    dataset = PixelArtDataset(
        load_dataset(config.dataset_name, split="train", cache_dir=config.hf_cache_dir),
        prompt=config.instance_prompt,
        size=config.resolution,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=1e-2)

    (
        unet,
        optimizer,
        loader,
        vae,
        text_encoder,
        text_encoder_2,
        quantizer,
    ) = accelerator.prepare(
        unet,
        optimizer,
        loader,
        vae,
        text_encoder,
        text_encoder_2,
        quantizer,
    )

    tokenizers = [tokenizer, tokenizer_2]
    text_encoders = [text_encoder, text_encoder_2]

    step = 0
    unet.train()
    while step < config.max_train_steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                images = batch["images"]
                batch_size = images.shape[0]

                with torch.no_grad():
                    prompt_embeds, pooled_embeds = encode_prompts(
                        batch["prompts"],
                        tokenizers,
                        text_encoders,
                        accelerator.device,
                    )
                    time_ids = make_time_ids(
                        batch_size,
                        config.resolution,
                        accelerator.device,
                        prompt_embeds.dtype,
                    )
                    added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": time_ids}

                    latents = vae.encode(images.to(vae.dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                pred = unet(
                    noisy,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                diff_loss = F.mse_loss(pred, noise)
                loss = diff_loss
                quant_reg_value = 0.0

                if config.w_quant_reg > 0:
                    low_noise = timesteps < config.quant_reg_max_t
                    if low_noise.any():
                        idx = low_noise.nonzero(as_tuple=True)[0]
                        x0_latents = predicted_x0(noisy[idx], pred[idx], timesteps[idx], noise_scheduler)
                        x0_image = vae.decode(x0_latents / vae.config.scaling_factor).sample.clamp(-1, 1)
                        x0_unit = (x0_image + 1) / 2
                        target = quantizer(x0_unit) * 2 - 1
                        quant_reg = F.mse_loss(x0_image, target)
                        loss = loss + config.w_quant_reg * quant_reg
                        quant_reg_value = quant_reg.item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % config.log_every == 0 and accelerator.is_main_process:
                logger.info("step=%5d diff=%.4f qreg=%.4f", step, diff_loss.item(), quant_reg_value)

            step += 1
            if step >= config.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_weights(accelerator.unwrap_model(unet), config.output_dir)

    accelerator.wait_for_everyone()
    return config.output_dir
