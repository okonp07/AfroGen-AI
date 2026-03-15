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

The exported artifact uses the model-repo filename expected by the app:

- `trained_backend_stub.json`

You can also populate hosted inference metadata at export time, for example:

```bash
python3 scripts/export_checkpoint_metadata.py \
  --hosted-model-id black-forest-labs/FLUX.1-schnell \
  --prompt-prefix "afrocentric studio portrait, ultra detailed" \
  --negative-prompt "blurry, distorted hands" \
  --guidance-scale 3.5 \
  --num-inference-steps 6
```

## Expected Use

1. Update the template values to match the real model repo contents.
2. Export the bundle.
3. Upload the resulting files to the Hugging Face model repo.
4. Switch the Space from `synthetic` to `hybrid` only after the backend validates as ready.

## Recommended Live Hybrid Flow

1. Provision a hosted text-to-image model on Hugging Face and note its model id.
2. Export the hybrid artifact bundle:

```bash
python3 scripts/export_checkpoint_metadata.py \
  --hosted-model-id black-forest-labs/FLUX.1-schnell \
  --prompt-prefix "afrocentric studio portrait, ultra detailed" \
  --negative-prompt "blurry, distorted hands" \
  --guidance-scale 3.5 \
  --num-inference-steps 6
```

3. Publish that bundle to the model repo:

```bash
HF_TOKEN=... python3 scripts/publish_hf_bundle.py --bundle-dir outputs/checkpoint_metadata_bundle
```

4. Set the Space variables:

- `AFROGEN_BACKEND=hybrid`
- `AFROGEN_ARTIFACT_PATH=hf://okonp007/afrogen-models/trained_backend_stub.json`

5. Test the live app. Once those steps are done, the Space is ready for real hosted-image generation testing.
