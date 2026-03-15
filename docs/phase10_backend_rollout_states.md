# Phase 10 Backend Rollout States

## Goal

Make cloud rollout safer by distinguishing backend lifecycle states clearly.

## States

### `missing`

No artifact metadata could be resolved.

### `stub`

The artifact exists, but it is still a scaffold.

### `metadata_only`

The artifact exists but does not declare a checkpoint path yet.

### `checkpoint_missing`

The artifact declares a checkpoint path, but the referenced checkpoint is not available.

### `ready_for_inference`

Artifact metadata exists and the referenced checkpoint path resolves successfully.

### `incomplete_inference_contract`

Checkpoint exists, but required hybrid inference metadata is still incomplete.

### `latent_editor_missing`

Latent editing is declared, but the latent editor asset cannot be resolved.

## Why This Helps

This gives the Space clearer behavior during rollout:

- deployment issues are easier to diagnose
- artifact problems are separated from app problems
- switching from `synthetic` to `hybrid` becomes safer
