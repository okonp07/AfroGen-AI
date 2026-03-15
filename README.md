---
title: AfroGen-AI
emoji: 🎨
colorFrom: yellow
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# AfroGen-AI


- APP LINK: [https://huggingface.co/spaces/okonp007/AfroGen-AI](https://huggingface.co/spaces/okonp007/AfroGen-AI)

AfroGen-AI is a cloud-deployed prompt-to-portrait app for afrocentric image generation with editable latent controls.

This repository now powers the live Hugging Face Space frontend:

- Frontend: [https://huggingface.co/spaces/okonp007/AfroGen-AI](https://huggingface.co/spaces/okonp007/AfroGen-AI)
- GitHub repo: [https://github.com/okonp07/AfroGen-AI](https://github.com/okonp07/AfroGen-AI)
- Model repo: [https://huggingface.co/okonp007/afrogen-models](https://huggingface.co/okonp007/afrogen-models)

The project started as a structured MVP scaffold and now includes a live hosted hybrid inference path backed by a Hugging Face model artifact.

## Current Live State

The app is currently deployed and working on Hugging Face Spaces.

- the frontend is live on Hugging Face Spaces
- the app is configured to use the `hybrid` backend
- backend artifact resolution runs through `hf://okonp007/afrogen-models/trained_backend_stub.json`
- the hybrid backend supports hosted Hugging Face image generation through `hosted_model_id`
- local latent-driven edits are still applied after hosted generation
- synthetic fallback remains available if hosted inference fails

At the moment, the live rollout is focused on proving hosted model-backed generation and preserving the controllable latent editing workflow end to end.

## Current MVP

The current MVP includes:

- prompt parsing for portrait attributes such as age, skin tone, hairstyle, mood, and accessories
- a deterministic latent control matrix that can be edited in the UI
- a synthetic portrait renderer that updates as the latent matrix changes
- a hosted Hugging Face hybrid backend path for real model-backed generation
- a Gradio app entrypoint for Hugging Face Spaces deployment
- a Streamlit app for local development
- a dataset manifest pipeline for training-ready metadata
- backend abstractions so a trained model can replace the synthetic backend cleanly
- a defined phase 3 research dataset slice and model strategy
- cloud-friendly artifact resolution for Hugging Face model repos

## Project Structure

```text
AfroGen-AI/
  app/                  Streamlit entrypoints
  configs/              App and model configuration
  data/                 Dataset staging folders
  docs/                 Architecture and roadmap docs
  models/               Placeholder for checkpoints
  outputs/              Generated images and exports
  scripts/              Helper scripts
  src/afrogen/          Python package
  tests/                Basic verification tests
```

## Quick Start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the local Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

4. Run the Gradio app used by Hugging Face Spaces:

```bash
python3 app.py
```

5. Build a training manifest from local images:

```bash
python3 scripts/prepare_dataset.py
```

6. Inspect the chosen phase 3 strategy:

```bash
python3 scripts/phase3_strategy.py
python3 scripts/training_readiness.py
```

7. Register a first curated batch and create a training stub:

```bash
python3 scripts/register_curation_batch.py
python3 scripts/build_training_stub.py
```

8. Inspect backend artifact loading:

```bash
python3 scripts/inspect_backend_artifact.py
```

9. Check Hugging Face Spaces readiness:

```bash
python3 scripts/space_readiness.py
```

10. Export the Hugging Face model-repo bundle:

```bash
python3 scripts/export_model_repo_bundle.py
```

11. Print the final Hugging Face publish handoff:

```bash
python3 scripts/publish_handoff.py
```

12. Inspect backend rollout lifecycle states:

```bash
python3 scripts/inspect_backend_artifact.py
```

13. Export a metadata-complete checkpoint bundle:

```bash
python3 scripts/export_checkpoint_metadata.py \
  --hosted-model-id black-forest-labs/FLUX.1-schnell
```

14. Inspect the live Hugging Face Space:

```bash
HF_TOKEN=... python3 scripts/check_live_space.py
```

15. Publish the current model bundle to Hugging Face:

```bash
HF_TOKEN=... python3 scripts/publish_hf_bundle.py
```

16. For live hybrid rollout, export and publish a hosted-model artifact:

```bash
python3 scripts/export_checkpoint_metadata.py \
  --hosted-model-id black-forest-labs/FLUX.1-schnell
HF_TOKEN=... python3 scripts/publish_hf_bundle.py --bundle-dir outputs/checkpoint_metadata_bundle
```

## Phase 2 Foundation

Phase 2 adds the bridge toward a real model project:

- `scripts/prepare_dataset.py` scans `data/raw/` and writes `data/processed/manifest.jsonl`
- `src/afrogen/data/` defines the dataset schema and prompt generation utilities
- `src/afrogen/backends/` defines the inference backend contract
- the Streamlit app now talks to a backend factory instead of hardcoding the synthetic pipeline

## Phase 3 Direction

The first real research baseline is now explicitly defined in the repo:

- dataset slice: `phase3_research_v1`
- priority sources: FairFace, FFHQ, BUPT-Balancedface
- target backend: `hybrid`
- model strategy: `latent-diffusion-plus-editor`
- baseline family: `sdxl-lora-plus-latent-editor`

## Stage 4 Foundation

Stage 4 starts the executable training path:

- curation batches can be registered into a slice registry
- manifests now carry `slice_name`, `source_dataset`, and `curation_batch`
- a training run plan can be generated from the current manifest
- a backend artifact stub can be written to `models/` for the trained backend contract

## Why This Version Matters

This repo now acts as a real foundation for the full product:

- reproducible package layout
- deterministic generation pipeline
- testable latent editing logic
- local Streamlit UX for prompt-based image generation
- live Hugging Face Space deployment with hosted hybrid inference

The next major milestone is improving generation quality with a stronger hosted model, better prompt conditioning, and future fine-tuned afrocentric checkpoints.

## Hugging Face Spaces

This repo now runs on Hugging Face Spaces with:

- root-level `app.py` Gradio entrypoint
- README front matter for Spaces configuration
- support for backend artifact paths that point to local files or Hugging Face Hub references
- env overrides for Space variables like `AFROGEN_BACKEND` and `AFROGEN_ARTIFACT_PATH`
- a live frontend at [https://huggingface.co/spaces/okonp007/AfroGen-AI](https://huggingface.co/spaces/okonp007/AfroGen-AI)

Supported artifact formats:

- local path like `models/trained_backend_stub.json`
- Hub path like `hf://username/repo-name/path/to/trained_backend_stub.json`

Hosted hybrid inference is now implemented in the repo for the `hybrid` backend. When a ready artifact includes `hosted_model_id`, AfroGen calls the hosted Hugging Face model and then applies local latent-driven image adjustments so the editor still influences the final portrait. If hosted inference fails, the backend falls back to the synthetic renderer.
