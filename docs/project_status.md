# Project Status

## Snapshot

AfroGen-AI has moved from a notebook-only research repo into a structured, deployed cloud project scaffold.

## What Is Done

### Core Repo Foundation

- project restructured into a reproducible Python package
- local Streamlit UI added
- Gradio `app.py` added for Hugging Face Spaces
- prompt parsing, latent matrix controls, and synthetic portrait rendering implemented
- backend abstraction layer added

### Data And Training Scaffolding

- dataset manifest pipeline implemented
- curation batch registry implemented
- training plan and training stub generation implemented
- model-repo bundle export implemented
- checkpoint-metadata bundle export implemented

### Backend Readiness

- trained backend artifact schema defined
- rollout states implemented
- hybrid inference readiness contract defined
- real hosted Hugging Face model-backed hybrid inference path implemented in the repo
- synthetic fallback remains available when hosted inference is unavailable or artifact metadata is incomplete

### Cloud Deployment

- Hugging Face Space created:
  `okonp007/AfroGen-AI`
- Hugging Face model repo created:
  `okonp007/afrogen-models`
- app pushed to the Hugging Face Space repo
- metadata-complete hosted hybrid artifact pushed to the Hugging Face model repo
- Space build issues fixed for:
  - README YAML metadata
  - `gradio` compatibility
  - `huggingface_hub` compatibility
- Space updated to run the latest hosted hybrid backend code
- `HF_TOKEN` configured as a Space secret for hosted inference

## Current Live Resources

- GitHub repo:
  `https://github.com/okonp07/AfroGen-AI`
- Hugging Face Space:
  `https://huggingface.co/spaces/okonp007/AfroGen-AI`
- Hugging Face model repo:
  `https://huggingface.co/okonp007/afrogen-models`

## Current Live Runtime Mode

The live Space is currently running as:

- backend: `hybrid`
- artifact path: `hf://okonp007/afrogen-models/trained_backend_stub.json`
- hosted model id: `black-forest-labs/FLUX.1-schnell`

The app now resolves a hosted hybrid backend artifact from the Hugging Face model repo and attempts real model-backed generation first, with synthetic fallback still available if hosted inference fails.

## Important Completed Milestones

- repo migrated off notebooks
- GitHub `main` contains the full scaffold history
- Hugging Face deployment path proven
- Space reached `RUNNING`

## What Still Remains

### Highest Priority

- rotate the Hugging Face token because it was pasted into chat
- continue testing live hybrid generation quality and prompt behavior
- monitor whether hosted inference consistently avoids synthetic fallback

### Next Engineering Milestone

- improve the quality and reliability of the hosted hybrid generation path
- evaluate stronger hosted model options or better prompt conditioning
- decide whether to keep using hosted inference or move toward a fine-tuned checkpoint path

### Real Model Work Still Needed

- curated dataset ingestion at meaningful scale
- actual model training or fine-tuning
- real checkpoint export
- optional latent editor asset export

## Recommended Next Phase

The next best phase is:

### Live Hybrid Preparation

- test live prompt-to-image generations on the Hugging Face Space frontend
- refine hosted model metadata and prompt defaults based on observed output quality
- reduce fallback frequency and improve production observability
- continue toward afrocentric fine-tuning and checkpoint-backed artifacts

## Repo Recovery Guidance

If context is lost, restart from:

1. `README.md`
2. `docs/project_status.md`
3. `docs/phase9_publish_handoff.md`
4. `docs/phase10_backend_rollout_states.md`
5. `docs/phase11_hybrid_inference_contract.md`
6. `docs/phase12_checkpoint_metadata_bundle.md`
