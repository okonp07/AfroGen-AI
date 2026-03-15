# Training Plan

## Phase 2 Outcome

The repo now includes:

- a dataset manifest pipeline
- backend abstractions for model inference
- a synthetic backend that behaves like a drop-in local model
- a placeholder trained backend interface

## Next Training Milestones

### 1. Dataset Curation

- collect afrocentric portrait imagery with clear rights and provenance
- align or crop faces consistently
- annotate attributes needed for prompt generation

### 2. Training Baseline

- start with a compact portrait generator baseline
- train or fine-tune using the generated manifest
- export checkpoints into `models/`

### 3. Inference Backend

- implement `TrainedAfroGenBackend`
- load checkpoint and tokenizer components
- keep the same `generate(prompt, seed, delta)` interface

### 4. Real-Time Editing

- map UI matrix edits into latent steering vectors
- preserve smooth response for interactive use
- expose interpretable controls for attributes later
