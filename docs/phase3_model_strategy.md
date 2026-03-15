# Phase 3 Model Strategy

## Chosen Direction

Use a hybrid architecture:

- text-to-image diffusion backbone for prompt fidelity
- editable latent controller for responsive real-time manipulation

## Why Hybrid

Pure diffusion is strong for prompts but weaker for responsive interactive editing.

Pure GAN editing is strong for latency and controllability but weaker for prompt alignment.

The hybrid path gives us the best product fit for AfroGen:

- prompt-driven generation
- editable latent matrix in the UI
- future attribute steering

## Phase 3 Baseline

### Backbone

Start with an SDXL-style latent diffusion backbone adapted through lightweight fine-tuning.

Recommended first baseline:

- LoRA fine-tune for afrocentric portrait domain adaptation
- training on the `phase3_research_v1` manifest

### Editable Control Space

Keep the current latent matrix UI and reinterpret it later as:

- a low-dimensional steering matrix
- or projected latent directions applied after prompt encoding / denoising setup

### Backend Contract

The trained backend should preserve:

`generate(prompt, seed, delta)`

Where:

- `prompt` is the user description
- `seed` controls reproducibility
- `delta` is the editable matrix from the UI

## Training Stages

### Stage 1

Domain-adapt the backbone on afrocentric portrait prompts.

### Stage 2

Learn a latent editor module or steering adapter that maps the UI matrix into controllable image changes.

### Stage 3

Optimize for interactive inference and expose higher-level semantic controls in Streamlit.

## Research Notes

The first backend implementation does not need perfect real-time inference.

It needs:

- reproducible text-to-image output
- deterministic seed behavior
- a pathway for matrix-based steering

That gives us a usable research backend before we optimize for speed.
