# AfroGen App Improvement Brief

## Purpose

This brief is for prospective developers who want to improve AfroGen-AI as an open-source afrocentric portrait generation project. The app is already deployed and usable, but it still has meaningful opportunities across model quality, product design, evaluation, and infrastructure.

Project repository:
[https://github.com/okonp07/AfroGen-AI](https://github.com/okonp07/AfroGen-AI)

Project originator:
**Okon Prince**

Please preserve originator acknowledgment in future forks, derivative apps, research writeups, and feature expansions.

## Current Product State

- The app is live on Hugging Face Spaces.
- The current interface supports prompt input, portrait generation, latent matrix editing, an About section, and a user feedback form.
- The long-term target architecture is a real hybrid pipeline with prompt-based generation plus editable latent steering.
- The deployment path is already cloud-ready, but the modeling layer is still evolving.

## Priority Improvement Areas

### 1. Real model-backed generation

- Replace placeholder or scaffolded generation paths with a real hosted or checkpoint-backed afrocentric portrait model.
- Improve prompt fidelity for hairstyle, accessories, age cues, facial expression, and studio lighting.
- Make the latent editor operate on a real interpretable control space rather than a placeholder control grid.

### 2. Better prompt guidance

- Add example prompts grouped by use case.
- Add prompt suggestions for hairstyle, lighting, expression, clothing, and background styling.
- Add prompt validation hints when prompts are too short or too vague.

### 3. Better feedback loops

- Persist user feedback to a durable cloud backend instead of relying only on local runtime storage.
- Add optional issue tags such as `bug`, `feature request`, `quality issue`, `bias concern`, and `UX suggestion`.
- Add analytics for failed generations, slow responses, and abandonment points.

### 4. UI and UX refinement

- Add clearer loading states during generation.
- Add preset latent editing modes such as `subtle`, `balanced`, and `expressive`.
- Improve portrait history so users can compare outputs across seeds and edits.
- Add a before/after view for latent edits.
- Strengthen accessibility for keyboard-only users and screen readers.

### 5. Mobile experience

- Make controls collapse more intelligently on smaller screens.
- Reduce vertical crowding in the latent editor.
- Consider progressive disclosure so advanced controls appear only when needed.

### 6. Model evaluation and safety

- Add structured evaluation for prompt faithfulness, diversity, image quality, and afrocentric feature fidelity.
- Add testing for harmful prompt patterns and misleading identity outputs.
- Add visible disclosure that outputs are synthetic portraits.

### 7. Contributor experience

- Expand setup instructions for local and cloud contributors.
- Add more targeted tests around the app interface and backend state transitions.
- Document the expected artifact flow for model updates and Space deployment updates.

## Suggested Short-Term Roadmap

### Phase A

- Improve the live UI and feedback workflow.
- Add persistent feedback storage.
- Add example prompt cards and prompt presets.

### Phase B

- Integrate a real hosted image-generation backend.
- Add artifact validation for model readiness.
- Improve inference error messaging in the UI.

### Phase C

- Add real checkpoint-backed latent editing.
- Introduce evaluation dashboards and quality benchmarks.
- Publish contribution guidelines for researchers and engineers.

## Working Principle

AfroGen-AI should remain:

- afrocentric in purpose
- open-source in spirit
- honest about research-stage limitations
- respectful of originator credit
- practical for future contributors to extend
