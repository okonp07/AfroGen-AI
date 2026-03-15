# Architecture Overview

## MVP Goal

Deliver a local, reproducible prototype that:

- accepts a portrait prompt
- generates a synthetic afrocentric face concept
- exposes an editable latent matrix
- updates the portrait after matrix edits

## MVP Components

### 1. Prompt Interpreter

Extracts lightweight semantic hints from user text:

- age group
- skin tone
- hairstyle
- expression
- accessories

### 2. Latent Controller

Builds a deterministic latent matrix from:

- prompt text
- random seed
- optional user edits

### 3. Portrait Renderer

Transforms prompt attributes and latent values into a stylized portrait illustration.

This renderer is a stand-in for a future trained generator.

### 4. Streamlit UI

Provides:

- prompt input
- seed input
- latent matrix editor
- instant regeneration
- export support

## Planned Upgrade Path

The renderer interface is intentionally narrow so we can later swap in:

- a fine-tuned diffusion model for prompt fidelity
- a GAN or latent editor for responsive manipulation
- or a hybrid diffusion-plus-latent workflow
