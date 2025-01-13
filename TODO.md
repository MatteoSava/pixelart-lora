# TODO

## Training
- [TODO] Run a one-step Modal smoke train after the `huggingface-secret` secret is configured
- [TODO] Tune `--w-quant-reg` only if generated samples drift from the six-level palette
- [TODO] Add a sampling script that loads the saved LoRA and exports preview images

## Operations
- [TODO] Record expected runtime and Modal GPU cost after the first full run
- [TODO] Add a download command for the exact best checkpoint once a training run is selected

## Demo
- [TODO] Add gallery persistence for selected Gradio outputs
- [TODO] Add a Modal web deployment if the demo should run without a local Gradio process

## Documentation
- [TODO] Add a script to regenerate README visual assets

## Notes
- Dataset source: `mattsava/nob`
- Output volume: `jalb-lora-outputs`
- HF cache volume: `jalb-hf-cache`
- LoRA demo checkpoint: `mattsava/rob-lora-checkpoint-2500`
- The ROB LoRA uses FLUX-style `transformer.*` weights, not SDXL `unet.*` weights
- The demo loads transformer LoRA weights only; text-encoder LoRA keys trigger a Diffusers PEFT conversion error
