from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    pretrained_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    dataset_name: str = "mattsava/nob"
    instance_prompt: str = "a low resolution, grayscale pixel art of JALB a pixel art cute character"
    output_dir: str = "./jalb-lora-v2"
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 400
    lora_rank: int = 4
    num_workers: int = 4
    mixed_precision: str = "bf16"
    log_every: int = 10
    w_quant_reg: float = 0.0
    quant_reg_max_t: int = 200
    seed: int = 42
    hf_cache_dir: str | None = None

    def validate(self) -> None:
        if self.resolution <= 0 or self.resolution % 8 != 0:
            raise ValueError("resolution must be a positive multiple of 8")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")
        if self.w_quant_reg < 0:
            raise ValueError("w_quant_reg cannot be negative")
        if self.quant_reg_max_t < 0:
            raise ValueError("quant_reg_max_t cannot be negative")
