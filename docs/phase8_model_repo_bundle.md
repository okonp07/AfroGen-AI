# Phase 8 Model Repo Bundle

## Goal

Prepare a clean bundle that can be uploaded to the Hugging Face model repo:

`okonp07/afrogen-models`

## Bundle Contents

- `trained_backend_stub.json`
- `training_run_plan.json`
- `README.md`

Later, this same bundle can also include:

- real checkpoint metadata
- checkpoint file references
- latent editor metadata

## Why A Bundle

The Space should consume one predictable remote artifact source.

A bundle makes that easier because:

- metadata stays versioned together
- upload steps are consistent
- the repo can evolve from stub to real model without changing the app contract

## Export Command

```bash
python3 scripts/export_model_repo_bundle.py
```

This writes a local staging directory under:

`outputs/model_repo_bundle/`

That directory is what you upload to the Hugging Face model repo.
