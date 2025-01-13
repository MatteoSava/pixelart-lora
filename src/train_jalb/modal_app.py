from __future__ import annotations

import modal

from train_jalb.config import TrainingConfig

APP_NAME = "train-jalb-sdxl-lora"
OUTPUT_VOLUME_NAME = "jalb-lora-outputs"
CACHE_VOLUME_NAME = "jalb-hf-cache"
OUTPUT_MOUNT = "/outputs"
CACHE_MOUNT = "/cache"
HF_SECRET_NAME = "huggingface-secret"

RUNTIME_DEPENDENCIES = [
    "accelerate==1.13.0",
    "datasets==4.8.5",
    "diffusers==0.37.1",
    "peft==0.19.1",
    "pillow==12.2.0",
    "safetensors==0.7.0",
    "torch==2.11.0",
    "torchvision==0.26.0",
    "tqdm==4.67.3",
    "transformers==5.8.0",
]

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*RUNTIME_DEPENDENCIES)
    .env(
        {
            "HF_HOME": CACHE_MOUNT,
            "HF_DATASETS_CACHE": f"{CACHE_MOUNT}/datasets",
            "HF_HUB_CACHE": f"{CACHE_MOUNT}/hub",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("train_jalb")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 8,
    volumes={OUTPUT_MOUNT: output_volume, CACHE_MOUNT: cache_volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def train_remote(config: dict) -> str:
    from train_jalb.config import TrainingConfig
    from train_jalb.train import train

    training_config = TrainingConfig(**config)
    output_dir = train(training_config)
    output_volume.commit()
    cache_volume.commit()
    return output_dir


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 20,
    scaledown_window=60 * 5,
    volumes={CACHE_MOUNT: cache_volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def generate_remote(payload: dict) -> dict:
    from train_jalb.inference import PipelineConfig, generate_png, request_from_payload

    result = generate_png(
        request_from_payload(payload),
        PipelineConfig(cache_dir=CACHE_MOUNT),
    )
    cache_volume.commit()
    return result


@app.local_entrypoint()
def main(
    dataset_name: str = "mattsava/nob",
    pretrained_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    output_name: str = "jalb-lora-v2",
    max_train_steps: int = 1500,
    learning_rate: float = 1e-4,
    mixed_precision: str = "bf16",
    resolution: int = 512,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    lora_rank: int = 4,
    num_workers: int = 4,
    w_quant_reg: float = 0.0,
    quant_reg_max_t: int = 200,
    seed: int = 42,
) -> None:
    config = TrainingConfig(
        pretrained_model=pretrained_model,
        dataset_name=dataset_name,
        output_dir=f"{OUTPUT_MOUNT}/{output_name}",
        resolution=resolution,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_train_steps=max_train_steps,
        lora_rank=lora_rank,
        num_workers=num_workers,
        mixed_precision=mixed_precision,
        w_quant_reg=w_quant_reg,
        quant_reg_max_t=quant_reg_max_t,
        seed=seed,
        hf_cache_dir=CACHE_MOUNT,
    )
    remote_output_dir = train_remote.remote(config.__dict__)
    print(f"Saved LoRA weights to Modal volume {OUTPUT_VOLUME_NAME}:{remote_output_dir}")
