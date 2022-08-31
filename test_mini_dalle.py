import torch
from PIL.Image import Image
from min_dalle import MinDalle
#!/usr/bin/env python
#\ coding: utf-8

model = MinDalle(
    models_root='./pretrained/',
    dtype=torch.float16,
    device='cuda',
    is_mega=False,
    is_reusable=True
)

image = model.generate_image(
    text='a woman face',
    seed=-1,
    grid_size=1,
    is_seamless=False,
    temperature=1,
    top_k=256,
    supercondition_factor=32,
    is_verbose=True
)

image.save('image.png')

print("Finished")