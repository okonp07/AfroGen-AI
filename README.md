# AfroGen-AI: A model for Generating High quality black faces

# AfroGen AI

AfroGen AI is a generative AI model based on Stable Diffusion that focuses on generating high-quality images of Black faces. The project utilizes a transformer-based model with a final layer trained exclusively on images of Black individuals. Additionally, the model incorporates a latent vector that is generated alongside the image, enabling seamless modifications to the image by adjusting the latent vector.

## Features
- **Black Face Generation**: Specializes in generating realistic and diverse images of Black people.
- **Latent Vector Editing**: Each generated image is accompanied by a latent vector, allowing users to fine-tune the image characteristics post-generation.
- **Transformer-Based Architecture**: Built upon a modified version of Stable Diffusion with an Afro-centric dataset.

## Installation

### Prerequisites
Ensure that you have the following dependencies installed:
- Python 3.8+
- PyTorch
- Hugging Face Diffusers
- Transformers
- OpenCV
- NumPy
- Matplotlib
- CUDA (for GPU acceleration)

To install the necessary libraries, run:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers opencv-python numpy matplotlib
```

## Training the Model

### 1. Data Preparation
Gather a dataset of Black faces with diverse attributes (gender, age, skin tone, expressions, etc.). Preprocess the images using OpenCV and store them in a structured format.

### 2. Fine-Tuning Stable Diffusion
Use the `diffusers` library to fine-tune a pre-trained Stable Diffusion model with the curated dataset:
```python
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel

# Load pre-trained model
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)

# Fine-tuning with Black face dataset (pseudo code)
pipeline.train_on_custom_dataset("path/to/black_faces_dataset")
```

### 3. Latent Vector Modification
Modify the latent space representation to ensure that generated images are tied to editable vectors:
```python
import torch

def generate_with_latent(model, prompt):
    latent_vector = torch.randn(1, 512)  # Example latent vector
    image = model(prompt, latent_vector)
    return image, latent_vector
```

## Usage

### Generating Images
Run the following to generate an image with AfroGen AI:
```python
image, latent_vector = generate_with_latent(pipeline, "A portrait of a Black woman with curly hair")
image.show()
```

### Modifying the Latent Vector
Adjust the latent vector to tweak facial attributes:
```python
new_latent = latent_vector + 0.1 * torch.randn_like(latent_vector)  # Slight variation
new_image = pipeline("A Black male with short hair", new_latent)
new_image.show()
```

## Roadmap
- **Phase 1**: Model fine-tuning on diverse Black face dataset 
- **Phase 2**: Implementing and refining latent vector modifications 
- **Phase 3**: Deploying AfroGen AI as an open-source model 

## License
This project is released under the MIT License.

## Contributors
- Okon Prince (Project Lead)
- Joseph Edet 
- Elizabeth Ajabor
- Ofigwe Hart
- Onyiobazi oquah 
- Edodi Christopher
- Michael Eti
- Austine Agbor 

## Acknowledgments
Special thanks to the open-source AI community for providing resources and frameworks that made this project possible.

