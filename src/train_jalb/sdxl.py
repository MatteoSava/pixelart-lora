from __future__ import annotations

import torch
from torch import nn


def encode_prompts(
    prompts: list[str],
    tokenizers: list,
    text_encoders: list[nn.Module],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SDXL prompt hidden states and pooled CLIP-bigG text embeddings."""
    embeds_list: list[torch.Tensor] = []
    pooled: torch.Tensor | None = None

    for tokenizer, text_encoder in zip(tokenizers, text_encoders, strict=True):
        tokens = tokenizer(
            prompts,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        output = text_encoder(tokens.input_ids.to(device), output_hidden_states=True)
        pooled = output[0]
        embeds_list.append(output.hidden_states[-2])

    if pooled is None:
        raise ValueError("at least one tokenizer and text encoder is required")

    return torch.cat(embeds_list, dim=-1), pooled


def make_time_ids(batch_size: int, resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """SDXL micro-conditioning: original size, crop top-left, target size."""
    return torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        device=device,
        dtype=dtype,
    ).expand(batch_size, -1)


def predicted_x0(
    noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    """Recover x0 from epsilon-parameterized DDPM predictions."""
    alphas = scheduler.alphas_cumprod.to(noisy_latents.device)
    alpha_t = alphas[timesteps].view(-1, 1, 1, 1).to(noisy_latents.dtype)
    return (noisy_latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
