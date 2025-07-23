import torch
import numpy as np
import math




def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

def split2(image_tensor, batch_size, h, w, patch_size=(256, 256)):
    """
    Splits a larger image tensor into patches for prediction.
    Assumes image_tensor is (1, C, H, W).
    """
    image_tensor = image_tensor.squeeze(0) # Remove batch dimension for easier slicing
    _, h_img, w_img = image_tensor.shape
    patches = []
    # Calculate overlap if needed to cover all areas
    stride_h = patch_size[0]
    stride_w = patch_size[1]

    for i in range(0, h_img, stride_h):
        for j in range(0, w_img, stride_w):
            patch = image_tensor[:, i:min(i + patch_size[0], h_img), j:min(j + patch_size[1], w_img)]
            # Pad if patch is smaller than patch_size (at edges)
            pad_h = patch_size[0] - patch.shape[1]
            pad_w = patch_size[1] - patch.shape[2]
            if pad_h > 0 or pad_w > 0:
                patch = nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0) # Pad with zeros or ones depending on context
            patches.append(patch)
    return torch.stack(patches) # Stack into a batch tensor
def merge_image2(patches_tensor, original_h, original_w, patch_size=(256, 256)):
    """
    Merges predicted patches back into a single image.
    Assumes patches_tensor is (N, C, H_patch, W_patch).
    """
    num_patches = patches_tensor.shape[0]
    # Calculate how many patches fit horizontally and vertically
    patches_per_row = math.ceil(original_w / patch_size[1])
    patches_per_col = math.ceil(original_h / patch_size[0])

    # Determine the padded image dimensions if split2 used padding
    padded_h = patches_per_col * patch_size[0]
    padded_w = patches_per_row * patch_size[1]

    # Initialize a blank image to reconstruct into
    reconstructed_image = torch.zeros(1, patches_tensor.shape[1], padded_h, padded_w, dtype=patches_tensor.dtype, device=patches_tensor.device)

    patch_idx = 0
    for i in range(patches_per_col):
        for j in range(patches_per_row):
            if patch_idx < num_patches:
                patch = patches_tensor[patch_idx]
                h_start = i * patch_size[0]
                w_start = j * patch_size[1]
                h_end = h_start + patch.shape[1]
                w_end = w_start + patch.shape[2]
                reconstructed_image[:, :, h_start:h_end, w_start:w_end] = patch
                patch_idx += 1

    return reconstructed_image[:, :, :original_h, :original_w].squeeze(0) # Remove batch dim and crop to original size



def getPatches(deg_image, clean_image, patch_size=(256, 256), mystride=128):
    """
    Extracts patches from degraded and clean images.
    Assumes deg_image and clean_image are 2D numpy arrays (grayscale).
    """
    h, w = deg_image.shape
    patches_wat = []
    patches_gt = []

    for i in range(0, h - patch_size[0] + 1, mystride):
        for j in range(0, w - patch_size[1] + 1, mystride):
            patch_wat = deg_image[i:i + patch_size[0], j:j + patch_size[1]]
            patch_gt = clean_image[i:i + patch_size[0], j:j + patch_size[1]]
            patches_wat.append(patch_wat)
            patches_gt.append(patch_gt)

    # Convert list of numpy arrays to a single numpy array, then to torch tensor
    # Add channel dimension
    patches_wat = np.array(patches_wat).reshape(-1, 1, patch_size[0], patch_size[1])
    patches_gt = np.array(patches_gt).reshape(-1, 1, patch_size[0], patch_size[1])

    # Normalize to [0, 1] if not already (plt.imread usually loads in [0,1])
    # Ensure float32 for PyTorch
    patches_wat = torch.from_numpy(patches_wat).float()
    patches_gt = torch.from_numpy(patches_gt).float()

    return patches_wat, patches_gt