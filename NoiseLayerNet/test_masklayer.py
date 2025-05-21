import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2


def load_mask(path):
    mask = Image.open(path).convert('L')
    mask = TF.to_tensor(mask)
    return mask


def single_affine_transform(mask):
    angle = random.uniform(-1, 1)
    translate = (
        random.uniform(0, 0.02) * mask.size(2),
        random.uniform(0, 0.02) * mask.size(1)
    )
    scale = random.uniform(0.95, 0.99)
    shear = random.uniform(-2, 2)

    transformed_mask = TF.affine(
        mask, angle=angle, translate=translate, scale=scale, shear=shear, fill=0
    )

    params = {
        'angle': angle,
        'translate': translate,
        'scale': scale,
        'shear': shear
    }

    return transformed_mask, params

def random_kernel_size1():
    choices = [3, 3, 3, 5, 7, 9]
    return random.choice(choices)

def random_kernel_size2():
    choices = [1]*12 + [3, 5, 7, 9]
    return random.choice(choices)


def mask_noise_layer(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    transformed_mask, params = single_affine_transform(mask)  # [B, C, H, W]

    transformed_mask_np = transformed_mask.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]

    final_masks = []
    for i in range(transformed_mask_np.shape[0]):
        single_mask = transformed_mask_np[i, :, :, 0]  # [H, W]

        kernel_size_1 = random_kernel_size2()
        blur1 = cv2.GaussianBlur(single_mask, (kernel_size_1, kernel_size_1), 0)

        binary_blur = (blur1 > 0).astype(np.float32)

        kernel_size_2 = random_kernel_size2()
        final_mask = cv2.GaussianBlur(binary_blur, (kernel_size_2, kernel_size_2), 0)

        final_mask = final_mask[..., np.newaxis]  # [H, W, C=1]

        final_mask = np.clip(final_mask, 0.0, 1.0)

        # final_mask = (final_mask > 0).astype(np.float32)

        final_masks.append(final_mask)

    final_masks_np = np.stack(final_masks, axis=0)  # [B, H, W, C=1]
    final_masks_tensor = torch.from_numpy(final_masks_np).permute(0, 3, 1, 2).to(mask.device)  # [B, C, H, W]

    return final_masks_tensor