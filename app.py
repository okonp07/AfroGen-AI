from __future__ import annotations

from pathlib import Path
import sys

import gradio as gr
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config
from afrogen.backends import create_backend


config = load_app_config()
app_config = config["app"]
latent_shape = tuple(app_config["latent_shape"])
backend = create_backend(
    name=app_config["backend"],
    image_size=app_config["image_size"],
    latent_shape=latent_shape,
    artifact_path=app_config["trained_backend_artifact"],
)


def _matrix_from_inputs(values: tuple[float, ...]) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape(latent_shape)


def generate_portrait(prompt: str, seed: float, *matrix_values: float):
    delta = _matrix_from_inputs(matrix_values)
    result = backend.generate(prompt=prompt, seed=int(seed), delta=delta)
    profile = {
        "age_group": result.profile.age_group,
        "skin_tone": result.profile.skin_tone,
        "hairstyle": result.profile.hairstyle,
        "expression": result.profile.expression,
        "accessory": result.profile.accessory,
        "backend": result.backend_name,
        "backend_message": result.backend_message,
    }
    return result.image, profile, result.base_latent.tolist(), result.edited_latent.tolist()


with gr.Blocks(theme=gr.themes.Soft(), title=app_config["title"]) as demo:
    gr.Markdown(f"# {app_config['title']}")
    gr.Markdown("Prompt-to-portrait MVP prepared for Hugging Face Spaces deployment.")
    gr.Markdown(f"Active backend: `{backend.info.name}` | Load state: `{backend.info.load_state}`")

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", lines=4, value=app_config["default_prompt"])
            seed = gr.Number(label="Seed", value=7, precision=0)
            generate = gr.Button("Generate")
            image_output = gr.Image(label="Generated Portrait", type="pil")
            profile_output = gr.JSON(label="Prompt Profile")
        with gr.Column(scale=1):
            gr.Markdown("## Latent Matrix Editor")
            sliders = []
            for row in range(latent_shape[0]):
                with gr.Row():
                    for col in range(latent_shape[1]):
                        sliders.append(
                            gr.Slider(
                                minimum=-1.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.0,
                                label=f"{row},{col}",
                            )
                        )
            base_matrix = gr.JSON(label="Base Latent Matrix")
            edited_matrix = gr.JSON(label="Edited Latent Matrix")

    generate.click(
        fn=generate_portrait,
        inputs=[prompt, seed, *sliders],
        outputs=[image_output, profile_output, base_matrix, edited_matrix],
    )


if __name__ == "__main__":
    demo.launch()
