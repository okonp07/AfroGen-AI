# Phase 3 Dataset Slice

## Slice Name

`phase3_research_v1`

## Purpose

Create the first curated research-grade dataset slice for afrocentric prompt-to-portrait generation.

This slice is not the final product dataset. Its job is to support:

- first text-to-image fine-tuning experiments
- prompt-template validation
- latent editing research
- backend integration work

## Target Composition

- `5,000` training images minimum
- `500` validation images minimum
- `250` test images minimum

## Priority Sources

1. FairFace
2. FFHQ
3. BUPT-Balancedface

## Filtering Goals

Keep images that are:

- high-resolution or clean enough for upscaling/cropping
- front-facing or near-front-facing portraits
- visually diverse across skin tones, age groups, and hairstyle categories
- suitable for prompt annotation

Exclude images that are:

- heavily occluded
- motion-blurred
- extreme-profile unless intentionally retained for variety
- duplicated or near-duplicated
- low-quality crops that would hurt identity consistency

## Attribute Coverage Targets

Try to balance across:

- age group: child, adult, senior
- gender presentation: woman, man, androgynous/unknown
- skin tone: warm, medium, deep
- hairstyle: afro, braids, locs, curly, headwrap, short-cropped
- expression: neutral, smile, calm, serious
- accessories: none, earrings, glasses, headwear
- lighting: studio, daylight, dramatic
- background: plain, outdoor, indoor

## Prompt Strategy

Each sample should have:

- one canonical prompt for training
- one shorter prompt for validation experiments
- structured metadata fields for future prompt regeneration

Canonical prompt example:

`A studio portrait of an adult Black woman with deep skin, braided hair, gold earrings, soft smile, and plain background`

## Curation Rules

- preserve provenance in metadata
- keep licensing notes per source
- avoid over-concentrating on one hairstyle or one demographic subgroup
- document every manual filtering pass

## Output

The slice should be represented as:

- `data/processed/manifest.jsonl`
- optional companion curation notes in `data/processed/`
