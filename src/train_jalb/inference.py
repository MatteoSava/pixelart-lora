from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from random import SystemRandom
from typing import Any

import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_LORA_REPO = "mattsava/rob-lora-checkpoint-2500"
DEFAULT_LORA_WEIGHT = "pytorch_lora_weights.safetensors"
DEFAULT_PROMPT = "ROB, a low resolution grayscale pixel art cute character, six gray levels, crisp pixel edges"
DEFAULT_NEGATIVE_PROMPT = "color, realistic, photo, 3d render, smooth gradients, blurry, noisy"

_RANDOM = SystemRandom()
_PIPELINE: FluxPipeline | None = None


@dataclass(frozen=True)
class PipelineConfig:
    base_model: str = DEFAULT_BASE_MODEL
    lora_repo: str = DEFAULT_LORA_REPO
    lora_weight_name: str = DEFAULT_LORA_WEIGHT
    cache_dir: str | None = None


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    width: int = 512
    height: int = 512
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    lora_scale: float = 1.0
    seed: int = -1

    def normalized_prompt(self) -> str:
        return self.prompt.strip() or DEFAULT_PROMPT

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError("width and height must be a multiple of 8")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be positive")
        if self.lora_scale < 0:
            raise ValueError("lora_scale cannot be negative")

    def resolved_seed(self) -> int:
        if self.seed >= 0:
            return self.seed
        return _RANDOM.randrange(0, 2**31)


def request_from_payload(payload: dict[str, Any]) -> GenerationRequest:
    return GenerationRequest(
        prompt=str(payload.get("prompt", DEFAULT_PROMPT)),
        negative_prompt=str(payload.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)),
        width=int(payload.get("width", 512)),
        height=int(payload.get("height", 512)),
        num_inference_steps=int(payload.get("num_inference_steps", 28)),
        guidance_scale=float(payload.get("guidance_scale", 3.5)),
        lora_scale=float(payload.get("lora_scale", 1.0)),
        seed=int(payload.get("seed", -1)),
    )


def filter_transformer_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if key.startswith("transformer.")}


def load_transformer_lora_state_dict(config: PipelineConfig) -> dict[str, torch.Tensor]:
    path = hf_hub_download(
        config.lora_repo,
        config.lora_weight_name,
        cache_dir=config.cache_dir,
    )
    state_dict = load_file(path, device="cpu")
    transformer_state_dict = filter_transformer_lora_state_dict(state_dict)
    if not transformer_state_dict:
        raise ValueError(f"No FLUX transformer LoRA weights found in {config.lora_repo}")
    return transformer_state_dict


def get_pipeline(config: PipelineConfig) -> FluxPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    pipe = FluxPipeline.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        cache_dir=config.cache_dir,
    )
    pipe.load_lora_weights(load_transformer_lora_state_dict(config), adapter_name="rob")
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    _PIPELINE = pipe
    return pipe


@torch.inference_mode()
def generate_png(request: GenerationRequest, pipeline_config: PipelineConfig) -> dict[str, Any]:
    request.validate()
    pipe = get_pipeline(pipeline_config)
    seed = request.resolved_seed()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=request.normalized_prompt(),
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        generator=generator,
        joint_attention_kwargs={"scale": request.lora_scale},
        max_sequence_length=512,
    ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return {"image": buffer.getvalue(), "seed": seed}


def image_from_png_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")
