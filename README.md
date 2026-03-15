# AfroGen-AI

AfroGen-AI is a reproducible MVP for generating synthetic afrocentric portrait concepts from text prompts and interactively editing a latent control matrix in a Streamlit UI.

This version replaces the earlier notebook-only experiments with a proper project structure that we can extend into a trained prompt-to-image system.

## Current MVP

The current MVP includes:

- prompt parsing for portrait attributes such as age, skin tone, hairstyle, mood, and accessories
- a deterministic latent control matrix that can be edited in the UI
- a synthetic portrait renderer that updates as the latent matrix changes
- a Streamlit app scaffold designed to accept a real model backend later
- a dataset manifest pipeline for training-ready metadata
- backend abstractions so a trained model can replace the synthetic backend cleanly
- a defined phase 3 research dataset slice and model strategy

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

3. Run the app:

```bash
streamlit run app/streamlit_app.py
```

4. Build a training manifest from local images:

```bash
python3 scripts/prepare_dataset.py
```

5. Inspect the chosen phase 3 strategy:

```bash
python3 scripts/phase3_strategy.py
python3 scripts/training_readiness.py
```

6. Register a first curated batch and create a training stub:

```bash
python3 scripts/register_curation_batch.py
python3 scripts/build_training_stub.py
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
- Streamlit UX for prompt-based image generation

The next major milestone is implementing the trained backend so the current app can switch from the synthetic renderer to a real afrocentric face model without changing the UI contract.
