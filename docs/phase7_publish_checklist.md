# Phase 7 Publish Checklist

## Assumed Repositories

- Space repo: `okonp007/AfroGen-AI`
- Model repo: `okonp007/afrogen-models`

## What Is Already Configured

- `README.md` contains Hugging Face Spaces YAML configuration
- `app.py` is the Gradio entrypoint
- the default artifact path points to:

`hf://okonp007/afrogen-models/trained_backend_stub.json`

## Step 1. Create the Hugging Face Space

Create a new Space on Hugging Face:

- SDK: `Gradio`
- Python version: `3.10`

This matches Hugging Face’s Spaces configuration model, which is driven from the YAML block at the top of the repository README.

## Step 2. Create the Model Repo

Create a model repository named:

`okonp007/afrogen-models`

Upload at least:

- `trained_backend_stub.json`

Later upload:

- checkpoint metadata
- model weights
- latent editor metadata

## Step 3. Set Space Variables and Secrets

In the Space settings:

- variable: `AFROGEN_BACKEND=synthetic` for the current demo
- later variable: `AFROGEN_BACKEND=hybrid`
- optional secret: `HF_TOKEN` if the model repo is private

Hugging Face documents using Space Settings for variables and secrets rather than hardcoding them in app code.

## Step 4. First Deployment Goal

For the first live deployment, use:

- backend: `synthetic`
- artifact path: default `hf://okonp07/afrogen-models/trained_backend_stub.json`

This proves:

- the Space boots
- Gradio renders
- Hub artifact resolution works

## Step 5. Switch to Model-Backed Inference Later

Once checkpoint metadata exists in the model repo:

- set `AFROGEN_BACKEND=hybrid`
- update the remote artifact metadata to `status: ready`
- point `checkpoint_path` to the hosted model artifact

## Official References

- Spaces config: [Hugging Face Spaces Configuration Reference](https://huggingface.co/docs/hub/en/spaces-config-reference)
- Hub downloads: [Download files from the Hub](https://huggingface.co/docs/huggingface_hub/guides/download)
- Spaces secrets and variables: [Spaces Overview](https://huggingface.co/docs/hub/en/spaces-overview)
