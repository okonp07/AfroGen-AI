# Stage 4 Execution

## Goal

Begin the real training path with concrete, repeatable mechanics.

Stage 4 does not train the final model yet. It creates the handoff between:

- curated source images
- a slice-aware manifest
- a training run plan
- a loadable backend artifact stub

## Workflow

### 1. Place curated images into a batch folder

Example:

- `data/raw/batches/fairface_batch_001/`
- `data/raw/batches/ffhq_batch_001/`

Each batch may include a `metadata.csv`.

### 2. Register the curation batch

```bash
python3 scripts/register_curation_batch.py
```

This writes:

- `data/processed/slice_registry.json`

### 3. Build the manifest

```bash
python3 scripts/prepare_dataset.py
```

### 4. Review readiness

```bash
python3 scripts/training_readiness.py
```

### 5. Generate the first training stub

```bash
python3 scripts/build_training_stub.py
```

This writes:

- `models/training_run_plan.json`
- `models/trained_backend_stub.json`

## Why This Matters

This gives us a consistent way to move from raw curation into backend implementation without hardcoding one-off training assumptions.
