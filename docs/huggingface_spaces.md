# Hugging Face Spaces Deployment

## Deployment Target

This repo is configured for Gradio on Hugging Face Spaces.

The root `README.md` includes the Spaces YAML block:

- `sdk: gradio`
- `app_file: app.py`

## App Entry Point

- `app.py` is the Spaces entrypoint
- `app/streamlit_app.py` remains the local development UI

## Artifact Resolution

The trained backend artifact can now be provided as:

- a local path
- or a Hugging Face Hub reference

Supported Hub format:

`hf://username/repo-name/path/to/trained_backend_stub.json`

When the app sees an `hf://` artifact path, it downloads the file with `hf_hub_download`.

## Recommended Cloud Layout

- GitHub repo for source
- Hugging Face Space for the app
- Hugging Face model repo for checkpoint and artifact metadata

Example:

- app repo: `okonp007/AfroGen-AI`
- model repo: `okonp007/afrogen-models`

The current repo config now defaults to:

`hf://okonp007/afrogen-models/trained_backend_stub.json`

## Suggested Secrets

If you use a private model repo, set:

- `HF_TOKEN`
- optionally `AFROGEN_ARTIFACT_PATH`
- optionally `AFROGEN_BACKEND`

## What To Upload To The Model Repo

- `trained_backend_stub.json`
- future checkpoint metadata
- future latent editor metadata

Later, once real weights exist, the backend artifact can point to:

- `checkpoint_path`
- `latent_editor_path`
