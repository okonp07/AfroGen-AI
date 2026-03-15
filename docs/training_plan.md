# Training Plan

## Phase 2 Outcome

The repo now includes:

- a dataset manifest pipeline
- backend abstractions for model inference
- a synthetic backend that behaves like a drop-in local model
- a placeholder trained backend interface

## Phase 3 Decision

The first real training path is now defined:

- dataset slice: `phase3_research_v1`
- target backend: `hybrid`
- model strategy: `latent-diffusion-plus-editor`
- baseline model family: `sdxl-lora-plus-latent-editor`

This means we are intentionally building toward:

- prompt-faithful generation through diffusion fine-tuning
- editable latent steering through a lightweight control module

## Next Training Milestones

### 1. Dataset Curation

- collect afrocentric portrait imagery with clear rights and provenance
- align or crop faces consistently
- annotate attributes needed for prompt generation
- build the first curated slice described in `docs/phase3_dataset_slice.md`

### 2. Training Baseline

- start with an SDXL-style LoRA baseline
- train or fine-tune using the generated manifest
- export checkpoints into `models/`

### 3. Inference Backend

- implement `TrainedAfroGenBackend`
- load checkpoint and tokenizer components
- keep the same `generate(prompt, seed, delta)` interface
- make `delta` the hook for the future latent editor

### 4. Real-Time Editing

- map UI matrix edits into latent steering vectors
- preserve smooth response for interactive use
- expose interpretable controls for attributes later
