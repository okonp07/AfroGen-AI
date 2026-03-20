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
author_image = PROJECT_ROOT / "assets" / "pic1.png"
backend = create_backend(
    name=app_config["backend"],
    image_size=app_config["image_size"],
    latent_shape=latent_shape,
    artifact_path=app_config["trained_backend_artifact"],
)


ABOUT_PROJECT_MARKDOWN = """
## About AfroGen-AI

AfroGen-AI is an open-source research and product prototype for prompt-driven afrocentric portrait generation. The long-term goal is to turn natural-language descriptions into realistic synthetic portraits while also exposing a controllable latent editing system that lets users shape the generated result in a transparent, interactive way.

### What the solution is trying to solve

Most general text-to-image systems do not give users enough control over afrocentric facial generation, hairstyle expression, cultural styling cues, or fine-grained portrait iteration. AfroGen-AI is designed to close that gap by combining:

- prompt-based portrait generation
- a latent control matrix for editing and steering
- a deployment-ready front end for real user interaction
- an open engineering foundation that other builders can extend

### How it works today

The current live version uses a safe synthetic backend for deployment validation. That means the app already proves the complete product workflow:

- a user writes a descriptive portrait prompt
- the app parses the prompt into key portrait attributes
- the app derives a latent matrix from the prompt and the seed
- the interface lets the user edit that matrix in real time
- the backend returns a generated portrait view plus interpretable metadata

At this stage, the public deployment is validating the user experience, backend contracts, cloud deployment path, and future hybrid architecture. The final target is a real checkpoint-backed hybrid model that replaces the placeholder renderer with realistic image generation.

### Open-source development

This project is open source and welcomes further developmental effort, research contribution, infrastructure support, UI improvements, and model-inference upgrades.

Official Git repository:
[https://github.com/okonp07/AfroGen-AI](https://github.com/okonp07/AfroGen-AI)

Project originator and lead:
**Okon Prince**

If you build on this work, extend the codebase, or contribute to future releases, the project originator should be clearly acknowledged.
"""


ABOUT_AUTHOR_MARKDOWN = """
## About the Author

**Okon Prince**  
Senior Data Scientist at MIVA Open University | AI Engineer & Data Scientist

I design and deploy end-to-end data systems that turn raw data into production-ready intelligence.

My core stack includes Python, Streamlit, BigQuery, Supabase, Hugging Face, PySpark, SQL, Machine Learning, LLMs, and Transformers.

My work spans risk scoring systems, A/B testing, Traditional and AI-powered dashboards, RAG pipelines, predictive analytics, LLM-based solutions and AI research.

Currently, I work as a Senior Data Scientist in the department of Research and Development at MIVA Open University, where I carry out AI / ML Research and build intelligent systems that drive analytics, decision support and scalable AI innovation.

**I believe: models are trained, systems are engineered and impact is delivered.**
"""


FOOTER_HTML = """
<div class="footer-note">
  <div>&copy; Okon Prince, 2026</div>
  <div>This project is based on the Mpeg-microbiome classification Zindi challenge.</div>
  <div>enquiries; okonp07@gmail.com</div>
</div>
"""


CUSTOM_CSS = """
:root {
  --afg-lilac: #d6c8f1;
  --afg-soft-lilac: #ebe4f8;
  --afg-ash: #d8d7de;
  --afg-smoke: #efeff4;
  --afg-purple: #5e3b8c;
  --afg-deep-purple: #2f153f;
  --afg-black: #121015;
  --afg-border: rgba(94, 59, 140, 0.18);
  --afg-panel: rgba(255, 255, 255, 0.82);
  --afg-panel-dark: rgba(30, 24, 38, 0.92);
  --afg-shadow: 0 18px 60px rgba(40, 24, 60, 0.14);
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(214, 200, 241, 0.85), transparent 32%),
    radial-gradient(circle at top right, rgba(193, 185, 221, 0.72), transparent 26%),
    linear-gradient(180deg, #f3f1f7 0%, #e8e6ee 52%, #dddbe3 100%);
  color: var(--afg-black);
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
  font-family: Georgia, "Times New Roman", serif;
  color: var(--afg-deep-purple);
  letter-spacing: -0.02em;
}

#app-shell {
  max-width: 1220px;
  margin: 0 auto;
  padding: 12px 8px 32px 8px;
}

.hero-card,
.panel-card,
.about-card,
.footer-wrap {
  border: 1px solid var(--afg-border);
  background: var(--afg-panel);
  box-shadow: var(--afg-shadow);
  border-radius: 28px;
  backdrop-filter: blur(12px);
}

.hero-card {
  padding: 14px 18px 8px 18px;
  margin-bottom: 16px;
}

.panel-card,
.about-card {
  padding: 18px;
}

#about-toggle,
#generate-btn,
#close-about {
  border-radius: 999px !important;
}

#about-toggle button,
#generate-btn button,
#close-about button {
  font-weight: 700;
  border: none !important;
}

#about-toggle button {
  background: linear-gradient(135deg, var(--afg-black), var(--afg-purple)) !important;
  color: white !important;
}

#generate-btn button {
  background: linear-gradient(135deg, var(--afg-purple), #8d63bf) !important;
  color: white !important;
}

#close-about button {
  background: rgba(94, 59, 140, 0.12) !important;
  color: var(--afg-deep-purple) !important;
}

#about-panel {
  margin-bottom: 16px;
}

#about-author-row {
  align-items: center;
}

#author-image {
  display: flex;
  justify-content: center;
  align-items: center;
}

#author-image img {
  width: min(300px, 100%);
  border-radius: 28px;
  border: 4px solid rgba(94, 59, 140, 0.18);
  box-shadow: 0 14px 40px rgba(38, 20, 51, 0.18);
}

#prompt-box textarea {
  min-height: 148px !important;
}

#result-image img {
  border-radius: 24px;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  margin-right: 10px;
  margin-bottom: 8px;
  border-radius: 999px;
  background: rgba(94, 59, 140, 0.09);
  border: 1px solid rgba(94, 59, 140, 0.18);
  color: var(--afg-deep-purple);
  font-size: 0.95rem;
  font-weight: 600;
}

.hero-subcopy {
  color: rgba(18, 16, 21, 0.78);
  font-size: 1rem;
  line-height: 1.65;
}

.footer-wrap {
  margin-top: 18px;
  padding: 18px 12px;
}

.footer-note {
  text-align: center;
  color: rgba(18, 16, 21, 0.8);
  font-size: 0.95rem;
  line-height: 1.8;
}

body.dark .gradio-container,
[data-theme="dark"] .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(126, 94, 167, 0.28), transparent 34%),
    radial-gradient(circle at top right, rgba(148, 128, 180, 0.2), transparent 28%),
    linear-gradient(180deg, #16121e 0%, #1d1825 55%, #26202f 100%);
  color: #f2eef8;
}

body.dark .hero-card,
body.dark .panel-card,
body.dark .about-card,
body.dark .footer-wrap,
[data-theme="dark"] .hero-card,
[data-theme="dark"] .panel-card,
[data-theme="dark"] .about-card,
[data-theme="dark"] .footer-wrap {
  background: var(--afg-panel-dark);
  border-color: rgba(214, 200, 241, 0.14);
}

body.dark .gradio-container h1,
body.dark .gradio-container h2,
body.dark .gradio-container h3,
[data-theme="dark"] .gradio-container h1,
[data-theme="dark"] .gradio-container h2,
[data-theme="dark"] .gradio-container h3 {
  color: #efe8fb;
}

body.dark .hero-subcopy,
body.dark .footer-note,
[data-theme="dark"] .hero-subcopy,
[data-theme="dark"] .footer-note {
  color: rgba(242, 238, 248, 0.82);
}

body.dark .status-pill,
[data-theme="dark"] .status-pill {
  background: rgba(214, 200, 241, 0.1);
  color: #efe8fb;
  border-color: rgba(214, 200, 241, 0.18);
}

@media (max-width: 900px) {
  #app-shell {
    padding: 6px 2px 24px 2px;
  }

  .hero-card,
  .panel-card,
  .about-card,
  .footer-wrap {
    border-radius: 22px;
  }

  #author-image img {
    width: min(240px, 100%);
    margin-top: 6px;
  }
}
"""


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


def toggle_about_panel(is_open: bool):
    new_state = not is_open
    return gr.update(visible=new_state), new_state


def close_about_panel():
    return gr.update(visible=False), False


with gr.Blocks(theme=gr.themes.Base(), title=app_config["title"], css=CUSTOM_CSS) as demo:
    about_open = gr.State(False)

    with gr.Column(elem_id="app-shell"):
        with gr.Column(elem_classes=["hero-card"]):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown(f"# {app_config['title']}")
                    gr.HTML(
                        """
                        <div class="hero-subcopy">
                          A cloud-ready portrait studio for afrocentric prompt generation, open-source research,
                          and future hybrid latent editing. The current live deployment proves the full UX while
                          the model-backed generation path continues to evolve.
                        </div>
                        """
                    )
                    gr.HTML(
                        f"""
                        <div>
                          <span class="status-pill">Backend: {backend.info.name}</span>
                          <span class="status-pill">Load State: {backend.info.load_state}</span>
                          <span class="status-pill">Rollout: {backend.info.rollout_state}</span>
                        </div>
                        """
                    )
                with gr.Column(scale=1, min_width=120):
                    about_button = gr.Button("About", elem_id="about-toggle")

        with gr.Column(visible=False, elem_id="about-panel", elem_classes=["about-card"]) as about_panel:
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown(ABOUT_PROJECT_MARKDOWN)
                with gr.Column(scale=1, min_width=140):
                    close_about = gr.Button("Close", elem_id="close-about")

            with gr.Row(elem_id="about-author-row"):
                with gr.Column(scale=3):
                    gr.Markdown(ABOUT_AUTHOR_MARKDOWN)
                with gr.Column(scale=2, elem_id="author-image"):
                    gr.Image(
                        value=str(author_image),
                        show_label=False,
                        interactive=False,
                        container=False,
                        elem_classes=["author-image-media"],
                    )

        with gr.Row():
            with gr.Column(scale=5, elem_classes=["panel-card"]):
                gr.Markdown("## Portrait Studio")
                prompt = gr.Textbox(
                    label="Describe the portrait you want to generate",
                    lines=5,
                    value=app_config["default_prompt"],
                    elem_id="prompt-box",
                )
                seed = gr.Number(label="Seed", value=7, precision=0)
                generate = gr.Button("Generate", elem_id="generate-btn")
                image_output = gr.Image(
                    label="Generated Portrait",
                    type="pil",
                    elem_id="result-image",
                )
                profile_output = gr.JSON(label="Prompt Profile")
            with gr.Column(scale=4, elem_classes=["panel-card"]):
                gr.Markdown("## Latent Matrix Editor")
                gr.Markdown(
                    "Fine-tune the latent control grid to steer the current portrait state. "
                    "This layout is optimized for both desktop and mobile interaction."
                )
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

        gr.HTML(FOOTER_HTML, elem_classes=["footer-wrap"])

    generate.click(
        fn=generate_portrait,
        inputs=[prompt, seed, *sliders],
        outputs=[image_output, profile_output, base_matrix, edited_matrix],
    )
    about_button.click(toggle_about_panel, inputs=about_open, outputs=[about_panel, about_open])
    close_about.click(close_about_panel, outputs=[about_panel, about_open])


if __name__ == "__main__":
    demo.launch()
