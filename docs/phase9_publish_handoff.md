# Phase 9 Publish Handoff

## Goal

Give AfroGen a final publish handoff for the current cloud deployment stage.

This phase answers three questions clearly:

1. What goes to the Hugging Face Space?
2. What goes to the Hugging Face model repo?
3. What environment settings should be applied first?

## Space Repo

Use the app repository:

- `okonp007/AfroGen-AI`

The Space should point at this repo and use:

- SDK: `Gradio`
- app entrypoint: `app.py`

## Model Repo

Use the model repository:

- `okonp007/afrogen-models`

Upload the contents of:

- `outputs/model_repo_bundle/`

That currently includes:

- `README.md`
- `trained_backend_stub.json`
- `training_run_plan.json`

## First Launch Settings

For the initial live test:

- `AFROGEN_BACKEND=synthetic`
- `AFROGEN_ARTIFACT_PATH=hf://okonp007/afrogen-models/trained_backend_stub.json`

Optional:

- `HF_TOKEN` only if the model repo is private

## What Success Looks Like

The first deployment is successful if:

- the Space builds without dependency errors
- the Gradio UI renders
- the prompt form works
- the app resolves the remote artifact path successfully
- the backend shows `synthetic` and the Space remains usable

## What Comes After

Once the cloud path is confirmed, the next real engineering milestone is:

- real model checkpoint metadata in the model repo
- `AFROGEN_BACKEND=hybrid`
- replacing placeholder inference with actual checkpoint-backed generation
