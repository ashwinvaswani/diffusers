from diffusers import StableDiffusionPipeline
import torch

model_id = "./saved_model/"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A DSLR photo of a sks person doing a thumbs up action wearing a red shirt, 8K"
negative_prompt="disfigured, deformed, duplicate, bad, immature, mutilated, cartoon, anime, 3d, painting, b&w, extra fingers, poorly drawn face, blurry"
image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=300, guidance_scale=15).images[0]

image.save("inference_output.png")