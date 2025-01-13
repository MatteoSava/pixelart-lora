from __future__ import annotations

import argparse

import gradio as gr
import modal

from train_jalb.inference import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, image_from_png_bytes
from train_jalb.modal_app import APP_NAME


def _remote_generator():
    return modal.Function.from_name(APP_NAME, "generate_remote")


def generate(
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance: float,
    lora_scale: float,
    width: int,
    height: int,
):
    result = _remote_generator().remote(
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "lora_scale": lora_scale,
            "width": width,
            "height": height,
        }
    )
    return image_from_png_bytes(result["image"]), f"seed={result['seed']}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ROB FLUX LoRA") as demo:
        gr.Markdown("# ROB FLUX LoRA")
        gr.Markdown("Grayscale pixel-art generation with FLUX and the `mattsava/rob-lora-checkpoint-2500` LoRA.")
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=3)
                negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2)
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=512, maximum=1024, step=64, value=512)
                    height = gr.Slider(label="Height", minimum=512, maximum=1024, step=64, value=512)
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=28)
                    guidance = gr.Slider(label="Guidance", minimum=1.0, maximum=8.0, step=0.5, value=3.5)
                with gr.Row():
                    lora_scale = gr.Slider(label="LoRA scale", minimum=0.0, maximum=1.5, step=0.05, value=1.0)
                    seed = gr.Number(label="Seed (-1 random)", value=-1, precision=0)
                button = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                output = gr.Image(label="Output", type="pil")
                seed_used = gr.Textbox(label="Run", interactive=False)

        button.click(
            fn=generate,
            inputs=[prompt, negative_prompt, seed, steps, guidance, lora_scale, width, height],
            outputs=[output, seed_used],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_demo().queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
