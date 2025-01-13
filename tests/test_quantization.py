import torch

from train_jalb.quantization import SixGrayQuantizer


def test_six_gray_quantizer_snaps_rgb_pixels_to_nearest_fixed_gray_level():
    quantizer = SixGrayQuantizer()
    image = torch.tensor(
        [
            [[0.01, 0.19], [0.41, 0.99]],
            [[0.01, 0.19], [0.41, 0.99]],
            [[0.01, 0.19], [0.41, 0.99]],
        ]
    )

    snapped = quantizer(image)

    assert snapped.shape == (3, 2, 2)
    assert torch.equal(snapped[0], torch.tensor([[0.0, 0.2], [0.4, 1.0]]))
    assert torch.equal(snapped[0], snapped[1])
    assert torch.equal(snapped[1], snapped[2])


def test_six_gray_quantizer_preserves_batch_shape_for_grayscale_input():
    quantizer = SixGrayQuantizer()
    image = torch.tensor([[[[0.11, 0.62]]], [[[0.79, 0.92]]]])

    snapped = quantizer(image)

    assert snapped.shape == (2, 3, 1, 2)
    assert torch.equal(snapped[:, 0], torch.tensor([[[0.2, 0.6]], [[0.8, 1.0]]]))
    assert torch.equal(snapped[:, 0], snapped[:, 1])
    assert torch.equal(snapped[:, 1], snapped[:, 2])
