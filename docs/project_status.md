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

### Cloud Deployment

- Hugging Face Space created:
  `okonp007/AfroGen-AI`
- Hugging Face model repo created:
  `okonp007/afrogen-models`
- app pushed to the Hugging Face Space repo
- initial model bundle pushed to the Hugging Face model repo
- Space build issues fixed for:
  - README YAML metadata
  - `gradio` compatibility
  - `huggingface_hub` compatibility

## Current Live Resources

- GitHub repo:
  `https://github.com/okonp07/AfroGen-AI`
- Hugging Face Space:
  `https://huggingface.co/spaces/okonp007/AfroGen-AI`
- Hugging Face model repo:
  `https://huggingface.co/okonp007/afrogen-models`

## Current Expected Runtime Mode

The live Space should currently be treated as:

- backend: `synthetic`

This is the correct safe state until a real hybrid checkpoint artifact is published and validated.

## Important Completed Milestones

- repo migrated off notebooks
- GitHub `main` contains the full scaffold history
- Hugging Face deployment path proven
- Space reached `RUNNING`

## What Still Remains

### Highest Priority

- test the live Space UI manually
- confirm `AFROGEN_BACKEND=synthetic` in Space settings
- rotate the Hugging Face token because it was pasted into chat

### Next Engineering Milestone

- publish a metadata-complete hybrid artifact to `okonp007/afrogen-models`
- then implement real checkpoint-backed inference in `src/afrogen/backends/trained.py`

### Real Model Work Still Needed

- curated dataset ingestion at meaningful scale
- actual model training or fine-tuning
- real checkpoint export
- optional latent editor asset export

## Recommended Next Phase

The next best phase is:

### Live Hybrid Preparation

- verify Space settings
- publish a metadata-complete artifact using the checkpoint metadata template
- keep backend on `synthetic` until the hybrid artifact validates cleanly
- then begin real checkpoint-backed inference implementation

## Repo Recovery Guidance

If context is lost, restart from:

1. `README.md`
2. `docs/project_status.md`
3. `docs/phase9_publish_handoff.md`
4. `docs/phase10_backend_rollout_states.md`
5. `docs/phase11_hybrid_inference_contract.md`
6. `docs/phase12_checkpoint_metadata_bundle.md`
