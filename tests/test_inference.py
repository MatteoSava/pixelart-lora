import pytest
import torch

from train_jalb.inference import (
    DEFAULT_LORA_REPO,
    DEFAULT_PROMPT,
    GenerationRequest,
    filter_transformer_lora_state_dict,
)


def test_default_lora_repo_uses_checkpoint_2500():
    assert DEFAULT_LORA_REPO == "mattsava/rob-lora-checkpoint-2500"


def test_generation_request_uses_project_defaults():
    request = GenerationRequest(prompt="  ")

    assert request.normalized_prompt() == DEFAULT_PROMPT
    assert request.width == 512
    assert request.height == 512
    assert request.num_inference_steps == 28
    assert request.guidance_scale == 3.5


def test_generation_request_rejects_invalid_dimensions():
    request = GenerationRequest(prompt="pixel art rob", width=513, height=512)

    with pytest.raises(ValueError, match="multiple of 8"):
        request.validate()


def test_generation_request_keeps_explicit_seed():
    request = GenerationRequest(prompt="pixel art rob", seed=123)

    assert request.resolved_seed() == 123


def test_filter_transformer_lora_state_dict_drops_text_encoder_keys():
    state_dict = {
        "transformer.block.lora_A.weight": torch.ones(1),
        "text_encoder.block.lora_A.weight": torch.ones(1),
    }

    filtered = filter_transformer_lora_state_dict(state_dict)

    assert list(filtered) == ["transformer.block.lora_A.weight"]
