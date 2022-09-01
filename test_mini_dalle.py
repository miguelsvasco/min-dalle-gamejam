import torch
from PIL.Image import Image
from min_dalle import MinDalle
import argparse
#!/usr/bin/env python
#\ coding: utf-8

# Init model - Mega requires 8GB VRAM minimum, float32 requires 12GB VRAM
model = MinDalle(
    models_root='./pretrained/',
    dtype=torch.float32,
    device='cuda',
    is_mega=True,
    is_reusable=True
)

# Parse commands
parser = argparse.ArgumentParser(description='Script to generate from Dall-E mini (mega version).')
parser.add_argument('--prompt', action='store', type=str,  help='the type of image to create (ANYTHING)')
parser.add_argument('--n',  type=int,  help='number of images to generate')
parser.add_argument('--seed', type=int, help='random seed to generate')

args = parser.parse_args()

for i in range(args.n):
    print(f'Generating image #{i}')
    image = model.generate_image(
        text=args.prompt,
        seed=args.seed + i,
        grid_size=1,
        is_seamless=False,
        temperature=1,
        top_k=256,
        supercondition_factor=32,
        is_verbose=True
    )
    image.save(f'results/image_{i}.png')

print("Finished")