# train-jalb

SDXL LoRA training pipeline for a grayscale pixel-art JALB character, backed by the public
[`mattsava/nob`](https://huggingface.co/datasets/mattsava/nob) dataset and executed on Modal GPUs.

The training data is quantized in the image pipeline to six fixed gray levels before VAE encoding. The UNet receives full
SDXL conditioning: both text encoders, concatenated penultimate hidden states, pooled CLIP-bigG embeddings, and SDXL time
IDs.

## Setup

Install and sync the pinned environment:

```bash
uv sync
```

Authenticate Modal:

```bash
uv run modal setup
```

Create a Hugging Face secret for Modal. This is recommended for SDXL downloads, especially if your account needs to
accept model terms:

```bash
uv run modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

## Train on Modal

```bash
uv run modal run src/train_jalb/modal_app.py \
  --dataset-name mattsava/nob \
  --output-name jalb-lora-v2 \
  --max-train-steps 1500 \
  --learning-rate 1e-4 \
  --mixed-precision bf16
```

Optional quantization regularizer:

```bash
uv run modal run src/train_jalb/modal_app.py \
  --output-name jalb-lora-v2-qreg \
  --max-train-steps 1500 \
  --w-quant-reg 0.1 \
  --quant-reg-max-t 200
```

Outputs are saved in the Modal volume `jalb-lora-outputs`, under the chosen output name:

```bash
uv run modal volume get jalb-lora-outputs /jalb-lora-v2 ./jalb-lora-v2
```

## Local Checks

```bash
uv run pytest
uv run ruff check .
```

## Gradio Demo

Deploy the Modal GPU functions:

```bash
uv run modal deploy src/train_jalb/modal_app.py
```

Launch the local Gradio UI:

```bash
uv run python -m train_jalb.gradio_app
```

Open `http://127.0.0.1:7860`. The UI calls the deployed `generate_remote` Modal function and uses the
`mattsava/rob-lora-checkpoint-2500` LoRA over FLUX. The checkpoint includes FLUX `transformer.*` weights and CLIP
text-encoder keys; the demo loads the transformer weights because the text-encoder keys hit a Diffusers PEFT conversion
edge case in the pinned stack.

## Project Layout

```text
src/train_jalb/
  config.py      Training configuration and validation
  data.py        Hugging Face dataset wrapper and collate function
  gradio_app.py  Local Gradio UI backed by Modal inference
  inference.py   SDXL LoRA generation helpers
  modal_app.py   Modal image, volumes, GPU function, and local entrypoint
  quantization.py
                 Six-level grayscale quantizer and pixel-art transform
  sdxl.py        SDXL prompt/time/x0 helper functions
  train.py       Framework-agnostic training loop
tests/
  test_quantization.py
  test_sdxl_helpers.py
```
