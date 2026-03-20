from __future__ import annotations

from datetime import datetime, timezone
import json
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
AUTHOR_IMAGE_URL = "https://raw.githubusercontent.com/okonp07/AfroGen-AI/main/assets/pic1.png"
GITHUB_REPO_URL = "https://github.com/okonp07/AfroGen-AI"
DEVELOPER_BRIEF_URL = f"{GITHUB_REPO_URL}/blob/main/docs/app_improvement_brief.md"
FEEDBACK_STORAGE_PATH = PROJECT_ROOT / "outputs" / "feedback" / "feedback_submissions.jsonl"
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
  <div>This project and its codebase are provided for research purposes only and are not intended for commercial use. It is distributed under the MIT License.</div>
  <div>For enquiries, please contact: okonp07@gmail.com</div>
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
#close-about,
#feedback-submit-btn {
  border-radius: 999px !important;
}

#about-toggle button,
#generate-btn button,
#close-about button,
#feedback-submit-btn button {
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

#feedback-submit-btn button {
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

.hero-subcopy {
  color: rgba(18, 16, 21, 0.78);
  font-size: 1rem;
  line-height: 1.65;
}

.hero-subcopy p {
  margin: 0 0 14px 0;
}

.deployment-note-box {
  margin-top: 8px;
  padding: 14px 16px;
  border-radius: 20px;
  background: rgba(94, 59, 140, 0.08);
  border: 1px solid rgba(94, 59, 140, 0.12);
}

.deployment-note-title {
  color: rgba(18, 16, 21, 0.74);
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.deployment-note-list {
  margin: 0;
  padding-left: 18px;
}

.deployment-note-list li {
  margin: 6px 0;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 16px;
}

.action-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  min-height: 56px;
  padding: 0 22px;
  border-radius: 999px;
  text-decoration: none !important;
  font-weight: 700;
  font-size: 0.92rem;
  color: var(--afg-deep-purple) !important;
  background: rgba(94, 59, 140, 0.09);
  border: 1px solid rgba(94, 59, 140, 0.18);
  box-shadow: 0 8px 22px rgba(47, 21, 63, 0.08);
  transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
  cursor: pointer;
}

.action-link::before {
  content: "";
  width: 18px;
  height: 18px;
  border-radius: 50%;
  flex: 0 0 18px;
  background: linear-gradient(135deg, #7b56ad, #2f153f);
  box-shadow: 0 0 0 6px rgba(94, 59, 140, 0.12);
}

.action-link:link,
.action-link:visited,
.action-link:hover,
.action-link:active,
.action-link:focus {
  color: var(--afg-deep-purple) !important;
  text-decoration: none !important;
}

.action-link:hover,
.action-link:focus-visible {
  transform: translateY(-1px);
  background: rgba(94, 59, 140, 0.13);
  box-shadow: 0 12px 28px rgba(47, 21, 63, 0.14);
}

.action-link:focus-visible {
  outline: 3px solid rgba(214, 200, 241, 0.75);
  outline-offset: 2px;
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

body.dark .deployment-note-box,
[data-theme="dark"] .deployment-note-box {
  background: rgba(214, 200, 241, 0.08);
  border-color: rgba(214, 200, 241, 0.12);
}

body.dark .deployment-note-title,
[data-theme="dark"] .deployment-note-title {
  color: rgba(242, 238, 248, 0.72);
}

body.dark .action-link,
[data-theme="dark"] .action-link,
body.dark .action-link:link,
body.dark .action-link:visited,
body.dark .action-link:hover,
body.dark .action-link:active,
body.dark .action-link:focus,
[data-theme="dark"] .action-link:link,
[data-theme="dark"] .action-link:visited,
[data-theme="dark"] .action-link:hover,
[data-theme="dark"] .action-link:active,
[data-theme="dark"] .action-link:focus {
  color: #efe8fb !important;
  background: rgba(214, 200, 241, 0.1);
  border-color: rgba(214, 200, 241, 0.18);
}

body.dark .action-link:hover,
body.dark .action-link:focus-visible,
[data-theme="dark"] .action-link:hover,
[data-theme="dark"] .action-link:focus-visible {
  background: rgba(214, 200, 241, 0.16);
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

  .hero-actions {
    flex-direction: column;
  }

  .action-link {
    width: 100%;
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


def submit_feedback(
    name: str,
    email: str,
    role: str,
    rating: float,
    feedback_text: str,
):
    cleaned_name = name.strip()
    cleaned_email = email.strip()
    cleaned_feedback = feedback_text.strip()
    if not cleaned_feedback:
        return (
            gr.update(
                value="Please enter your feedback before submitting the form.",
                visible=True,
            ),
            name,
            email,
            role,
            rating,
            feedback_text,
        )

    feedback_entry = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "name": cleaned_name or "Anonymous",
        "email": cleaned_email,
        "role": role,
        "rating": int(rating),
        "feedback": cleaned_feedback,
    }

    save_message = "Feedback saved successfully for review."
    try:
        FEEDBACK_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_STORAGE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(feedback_entry, ensure_ascii=True) + "\n")
    except OSError as exc:
        save_message = f"Feedback captured, but local storage failed: {exc}"

    thank_you = (
        f"Thank you{f' {cleaned_name}' if cleaned_name else ''}. "
        f"Your feedback has been recorded. {save_message}"
    )
    return (
        gr.update(value=thank_you, visible=True),
        "",
        "",
        "App user",
        4,
        "",
    )


with gr.Blocks(theme=gr.themes.Base(), title=app_config["title"], css=CUSTOM_CSS) as demo:
    about_open = gr.State(False)

    with gr.Column(elem_id="app-shell"):
        with gr.Column(elem_classes=["hero-card"]):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown(f"# {app_config['title']}")
                    gr.HTML(
                        f"""
                        <div class="hero-subcopy">
                          <p>
                            A cloud-ready portrait studio for afrocentric prompt generation, open-source research,
                            and future hybrid latent editing. The current live deployment proves the full UX while
                            the model-backed generation path continues to evolve.
                          </p>
                          <div class="deployment-note-box">
                            <div class="deployment-note-title">Deployment Notes</div>
                            <ul class="deployment-note-list">
                              <li><strong>Backend:</strong> {backend.info.name}</li>
                              <li><strong>Load state:</strong> {backend.info.load_state}</li>
                              <li><strong>Rollout readiness:</strong> {backend.info.rollout_state}</li>
                            </ul>
                          </div>
                          <div class="hero-actions" role="navigation" aria-label="Project actions">
                            <a class="action-link" href="{GITHUB_REPO_URL}" target="_blank" rel="noopener noreferrer">
                              Official GitHub Repo
                            </a>
                            <a class="action-link" href="{DEVELOPER_BRIEF_URL}" target="_blank" rel="noopener noreferrer">
                              Developer Brief
                            </a>
                            <a class="action-link" href="#feedback-panel">
                              User Feedback Form
                            </a>
                          </div>
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
                    gr.HTML(
                        f"""
                        <div class="author-image-media">
                          <img src="{AUTHOR_IMAGE_URL}" alt="Okon Prince portrait" />
                        </div>
                        """
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

        with gr.Column(elem_id="feedback-panel", elem_classes=["panel-card"]):
            gr.Markdown("## User Feedback Form")
            gr.Markdown(
                "Share product feedback, bugs, feature ideas, accessibility issues, or model-quality observations."
            )
            with gr.Row():
                feedback_name = gr.Textbox(label="Name", placeholder="Optional")
                feedback_email = gr.Textbox(label="Email", placeholder="Optional")
            with gr.Row():
                feedback_role = gr.Dropdown(
                    choices=["App user", "Researcher", "Developer", "Designer", "Other"],
                    value="App user",
                    label="You are",
                )
                feedback_rating = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=4,
                    label="Experience rating",
                )
            feedback_text = gr.Textbox(
                label="Your feedback",
                lines=5,
                placeholder="Tell us what worked, what did not, and what you would improve.",
            )
            feedback_submit = gr.Button("Submit Feedback", elem_id="feedback-submit-btn")
            feedback_status = gr.Markdown(visible=False)

        gr.HTML(FOOTER_HTML, elem_classes=["footer-wrap"])

    generate.click(
        fn=generate_portrait,
        inputs=[prompt, seed, *sliders],
        outputs=[image_output, profile_output, base_matrix, edited_matrix],
    )
    about_button.click(toggle_about_panel, inputs=about_open, outputs=[about_panel, about_open])
    close_about.click(close_about_panel, outputs=[about_panel, about_open])
    feedback_submit.click(
        fn=submit_feedback,
        inputs=[feedback_name, feedback_email, feedback_role, feedback_rating, feedback_text],
        outputs=[
            feedback_status,
            feedback_name,
            feedback_email,
            feedback_role,
            feedback_rating,
            feedback_text,
        ],
    )


if __name__ == "__main__":
    demo.launch()
