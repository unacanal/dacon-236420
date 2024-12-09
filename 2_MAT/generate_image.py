# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle with test time augmentation."""
import cv2
import pyspng
import glob
import os
import re
import random
from typing import List, Optional, Tuple

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import legacy
from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator


def apply_augmentation(image: torch.Tensor, mask: torch.Tensor, aug_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply augmentation to both image and mask."""
    if aug_type == 'original':
        return image, mask
    elif aug_type == 'flip_h':
        return torch.flip(image, [3]), torch.flip(mask, [3])
    elif aug_type == 'flip_v':
        return torch.flip(image, [2]), torch.flip(mask, [2])
    elif aug_type == 'rot90':
        return torch.rot90(image, 1, [2, 3]), torch.rot90(mask, 1, [2, 3])
    elif aug_type == 'rot180':
        return torch.rot90(image, 2, [2, 3]), torch.rot90(mask, 2, [2, 3])
    elif aug_type == 'rot270':
        return torch.rot90(image, 3, [2, 3]), torch.rot90(mask, 3, [2, 3])
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def reverse_augmentation(output: torch.Tensor, aug_type: str) -> torch.Tensor:
    """Reverse the augmentation applied to the output."""
    if aug_type == 'original':
        return output
    elif aug_type == 'flip_h':
        return torch.flip(output, [3])
    elif aug_type == 'flip_v':
        return torch.flip(output, [2])
    elif aug_type == 'rot90':
        return torch.rot90(output, -1, [2, 3])
    elif aug_type == 'rot180':
        return torch.rot90(output, -2, [2, 3])
    elif aug_type == 'rot270':
        return torch.rot90(output, -3, [2, 3])
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dpath', help='the path of the input image', required=True)
@click.option('--mpath', help='the path of the mask')
@click.option('--resolution', type=int, help='resolution of input image', default=512, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--tta', is_flag=True, help='Enable test time augmentation', default=True, show_default=True)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    dpath: str,
    mpath: Optional[str],
    resolution: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    tta: bool,
):
    """Generate images using pretrained network pickle with test time augmentation."""
    seed = 240  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Loading data from: {dpath}')
    img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg'))

    if mpath is not None:
        print(f'Loading mask from: {mpath}')
        mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg'))
        assert len(img_list) == len(mask_list), 'illegal mapping'

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    # Define augmentation types
    aug_types = ['original', 'flip_h', 'flip_v', 'rot90', 'rot180', 'rot270'] if tta else ['original']

    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        image = image.transpose(2, 0, 1) # HWC => CHW
        image = image[:3]
        return image

    if resolution != 512:
        noise_mode = 'random'
        
    with torch.no_grad():
        for i, ipath in enumerate(img_list):
            iname = os.path.basename(ipath).replace('.jpg', '.png')
            print(f'Processing: {iname}')
            image = read_image(ipath)
            image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)

            if mpath is not None:
                mask = cv2.imread(mask_list[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                mask = 1 - mask
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
            else:
                mask = RandomMask(resolution) # adjust the masking ratio by using 'hole_range'
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

            # Initialize accumulator for averaged output
            accumulated_output = None
            
            # Process each augmentation
            for aug_type in aug_types:
                # Apply augmentation
                aug_image, aug_mask = apply_augmentation(image, mask, aug_type)
                
                # Generate output with fixed random seed for consistency
                torch.manual_seed(seed)
                z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
                output = G(aug_image, aug_mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                
                # Reverse augmentation on the output
                output = reverse_augmentation(output, aug_type)
                
                # Accumulate output
                if accumulated_output is None:
                    accumulated_output = output
                else:
                    accumulated_output += output
            
            # Average the accumulated outputs
            final_output = accumulated_output / len(aug_types)
            
            # Convert to uint8 and save
            final_output = (final_output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            final_output = final_output[0].cpu().numpy()
            PIL.Image.fromarray(final_output, 'RGB').save(f'{outdir}/{iname}')


if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------