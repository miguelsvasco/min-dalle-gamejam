import torch
from PIL.Image import Image
from min_dalle import MinDalle
import argparse
#!/usr/bin/env python
#\ coding: utf-8


# Init model - Mega requires 8GB VRAM minimum
model = MinDalle(
    models_root='./pretrained/',
    dtype=torch.float16,
    device='cuda',
    is_mega=False,
    is_reusable=True
)

# Parse commands
parser = argparse.ArgumentParser(description='Script to generate from Dall-E mini (mega version).')
parser.add_argument('--prompt', action='store', type=str,  help='the type of image to create (ANYTHING)')
parser.add_argument('--n',  type=int,  help='number of images to generate')
parser.add_argument('--seed', type=int, help='random seed to generate')

args = parser.parse_args()

image = model.generate_image(
    text=args.prompt,
    seed=args.seed,
    grid_size=args.n,
    is_seamless=False,
    temperature=1,
    top_k=256,
    supercondition_factor=32,
    is_verbose=True
)

import ipdb; ipdb.set_trace()
image.save('image.png')

print("Finished")