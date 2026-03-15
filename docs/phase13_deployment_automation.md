# Phase 13 Deployment Automation

## Goal

Automate the two manual deployment paths that now exist:

- publishing the app repo to the Hugging Face Space repo
- publishing the model bundle to the Hugging Face model repo

## Why This Matters

We have already proven that:

- the Space can be created and run
- the model repo can host the artifact bundle

The next improvement is to make these updates repeatable.

## New Automation Scripts

### `scripts/check_live_space.py`

Inspects:

- Space runtime stage
- current error message
- current Space variables

### `scripts/publish_hf_bundle.py`

Publishes:

- `outputs/model_repo_bundle/` to the Hugging Face model repo

This uses `HF_TOKEN` and the configured Hugging Face namespace.

## Still Manual For Now

We are not yet fully automating Space repo mirroring from the GitHub repo because that is more invasive and easier to get wrong without a dedicated publish flow.

For now, the highest-value automation is:

- live status inspection
- model bundle publishing

## Required Environment

- `HF_TOKEN`
