# Phase 12 Checkpoint Metadata Bundle

## Goal

Create a repeatable path from a stub artifact to a metadata-complete hybrid artifact.

## Template

The repository now includes a template at:

- `models/hybrid_backend_artifact_template.json`

This is the starting point for the first real model-repo metadata package.

## What The Template Represents

It describes a model repo state where:

- a checkpoint path is declared
- prompt generation support is declared
- a scheduler name is declared
- latent editing can be declared later when the corresponding asset exists

## Export Command

```bash
python3 scripts/export_checkpoint_metadata.py
```

This writes a metadata-complete artifact bundle to:

- `outputs/checkpoint_metadata_bundle/`

## Expected Use

1. Update the template values to match the real model repo contents.
2. Export the bundle.
3. Upload the resulting files to the Hugging Face model repo.
4. Switch the Space from `synthetic` to `hybrid` only after the backend validates as ready.
