#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os # For checking file existence
from tqdm import tqdm

from utils import split2, merge_image2
from models.models import Generator

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if len(sys.argv) != 4:
    print("Usage: python predict.py <weights_path> <input_image_path> <output_image_path>")
    sys.exit(1)

weights_path = sys.argv[1]
deg_image_path = sys.argv[2]
save_path = sys.argv[3]

generator = Generator(input_channels=1, output_channels=1, biggest_layer=1024)
generator.to(device)

if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at {weights_path}")
    sys.exit(1)
try:
    generator.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Successfully loaded generator weights from {weights_path}")
except Exception as e:
    print(f"Error loading weights from {weights_path}: {e}")
    sys.exit(1)

generator.eval()

if not os.path.exists(deg_image_path):
    print(f"Error: Input image file not found at {deg_image_path}")
    sys.exit(1)

deg_image_pil = Image.open(deg_image_path).convert('L') 
deg_image_np = np.array(deg_image_pil).astype(np.float32) / 255.0 

h_orig, w_orig = deg_image_np.shape

h_pad = ((h_orig // 256) + (1 if h_orig % 256 != 0 else 0)) * 256
w_pad = ((w_orig // 256) + (1 if w_orig % 256 != 0 else 0)) * 256

test_padding_np = np.zeros((h_pad, w_pad), dtype=np.float32)
test_padding_np[:h_orig, :w_orig] = deg_image_np

test_image_tensor = torch.from_numpy(test_padding_np).unsqueeze(0).unsqueeze(0).to(device)

test_image_patches = split2(test_image_tensor, 1, h=h_pad, w=w_pad)

predicted_list = []
with torch.no_grad():
    for l in tqdm(range(test_image_patches.shape[0]), desc="Predicting patches"):
        patch_input = test_image_patches[l].unsqueeze(0).to(device) 
        
        predicted_patch_tensor = generator(patch_input)
        
        predicted_patch_np = predicted_patch_tensor.squeeze(0).squeeze(0).cpu().numpy()
        predicted_list.append(predicted_patch_np)

predicted_patches_stacked_np = np.array(predicted_list) 
predicted_patches_stacked_tensor = torch.from_numpy(predicted_patches_stacked_np).unsqueeze(1)

predicted_image_full_tensor = merge_image2(predicted_patches_stacked_tensor, h_orig, w_orig)


predicted_image_final_np = predicted_image_full_tensor.squeeze(0).cpu().numpy() 


print(f"Saving predicted image to {save_path}")
plt.imsave(save_path, predicted_image_final_np, cmap='gray')
print("Prediction complete.")