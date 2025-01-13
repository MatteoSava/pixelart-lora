from types import SimpleNamespace

import torch

from train_jalb.sdxl import encode_prompts, make_time_ids, predicted_x0


class FakeTokenizer:
    model_max_length = 4

    def __call__(self, prompts, **_kwargs):
        return SimpleNamespace(input_ids=torch.ones((len(prompts), self.model_max_length), dtype=torch.long))


class FakeTextOutput:
    def __init__(self, pooled, hidden):
        self._pooled = pooled
        self.hidden_states = [torch.zeros_like(hidden), hidden, torch.full_like(hidden, -1)]

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)
        return self._pooled


class FakeTextEncoder:
    def __init__(self, width: int, pooled_width: int):
        self.width = width
        self.pooled_width = pooled_width

    def __call__(self, input_ids, output_hidden_states: bool):
        assert output_hidden_states is True
        batch, seq_len = input_ids.shape
        hidden = torch.full((batch, seq_len, self.width), float(self.width), device=input_ids.device)
        pooled = torch.full((batch, self.pooled_width), float(self.pooled_width), device=input_ids.device)
        return FakeTextOutput(pooled, hidden)


def test_encode_prompts_concatenates_penultimate_states_and_uses_second_pooled_output():
    prompt_embeds, pooled_embeds = encode_prompts(
        ["jalb", "pixel art"],
        [FakeTokenizer(), FakeTokenizer()],
        [FakeTextEncoder(3, 5), FakeTextEncoder(7, 11)],
        torch.device("cpu"),
    )

    assert prompt_embeds.shape == (2, 4, 10)
    assert torch.all(prompt_embeds[..., :3] == 3.0)
    assert torch.all(prompt_embeds[..., 3:] == 7.0)
    assert pooled_embeds.shape == (2, 11)
    assert torch.all(pooled_embeds == 11.0)


def test_make_time_ids_matches_sdxl_micro_conditioning_layout():
    time_ids = make_time_ids(batch_size=2, resolution=512, device=torch.device("cpu"), dtype=torch.float32)

    assert time_ids.shape == (2, 6)
    assert time_ids.dtype == torch.float32
    assert time_ids.tolist() == [
        [512.0, 512.0, 0.0, 0.0, 512.0, 512.0],
        [512.0, 512.0, 0.0, 0.0, 512.0, 512.0],
    ]


def test_predicted_x0_recovers_clean_latents_from_epsilon_prediction():
    scheduler = SimpleNamespace(alphas_cumprod=torch.tensor([0.25, 0.81]))
    clean = torch.tensor([[[[0.5]]], [[[1.5]]]])
    noise = torch.tensor([[[[2.0]]], [[[3.0]]]])
    timesteps = torch.tensor([0, 1], dtype=torch.long)
    alpha = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    recovered = predicted_x0(noisy, noise, timesteps, scheduler)

    assert torch.allclose(recovered, clean)
