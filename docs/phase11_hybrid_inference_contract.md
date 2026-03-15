# Phase 11 Hybrid Inference Contract

## Goal

Define the minimum metadata contract for a real hybrid backend.

The model repo should not be considered inference-ready just because it contains a checkpoint path.

## Required For Hybrid Inference

- `checkpoint_path`
- `scheduler_name`
- `supports_prompt_generation=true`

## Required For Latent Editing Support

If `supports_latent_editing=true`, then:

- `latent_editor_path` must be declared
- the referenced asset must exist

## Optional But Useful

- `prompt_encoder_path`
- `manifest_path`
- `training_run_plan_path`

## Resulting Validation States

In addition to previous rollout states, the backend can now report:

### `incomplete_inference_contract`

Checkpoint exists, but the metadata is still missing required inference fields.

### `latent_editor_missing`

Latent editing is declared, but the editor asset is not available.
