import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random
import io

# Load Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

def generate_image(prompt):
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=50).images[0]
    return image

def main():
    print("Welcome to the Stable Diffusion App!")
    while True:
        prompt = input("Enter a text prompt (or type 'exit' to quit): ").strip()
        if prompt.lower() == 'exit':
            break
        print("Generating your image...")
        image = generate_image(prompt)
        image.show()

if __name__ == "__main__":
    main()
