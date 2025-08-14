# train.py (PyTorch Equivalent)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import random
import imageio

from models.models import Generator, Discriminator, GAN, get_optimizer
from utils import getPatches, split2, merge_image2, psnr

from torch.utils.data import Dataset, DataLoader


class PatchDataset(Dataset):
    def __init__(self, deg_patches, clean_patches):
        self.deg_patches = deg_patches
        self.clean_patches = clean_patches

    def __len__(self):
        return len(self.deg_patches)

    def __getitem__(self, idx):
        return self.deg_patches[idx], self.clean_patches[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_size = (256, 256, 1)
SCANNED_PATH = "data/train/scanned/"
GROUND_TRUTH_PATH = "data/train/ground_truth/"
VALIDATION_CLEAN_PATH = 'CLEAN/VALIDATION/GT/'
VALIDATION_DATA_PATH = 'CLEAN/VALIDATION/DATA/'

criterion_discriminator = nn.MSELoss()

criterion_generator_pixel = nn.BCELoss()
criterion_generator_gan = nn.MSELoss()


def train_gan(generator, discriminator, ep_start=1, epochs=1, batch_size=1):
    generator.to(device)
    discriminator.to(device)

    optimizer_G = get_optimizer(generator)
    optimizer_D = get_optimizer(discriminator)

    list_deg_images = [f
                       for f in os.listdir(SCANNED_PATH)
                       if f.endswith(".png")]
    list_clean_images = [f
                         for f in os.listdir(GROUND_TRUTH_PATH)
                         if f.endswith(".png")]

    list_deg_images.sort()
    list_clean_images.sort()

    # Preload all patches
    all_wat_patches = []
    all_gt_patches = []

    for im_idx in tqdm(range(len(list_deg_images))):
        deg_image_path = os.path.join(SCANNED_PATH,
                                      list_deg_images[im_idx])
        clean_image_path = os.path.join(GROUND_TRUTH_PATH,
                                        list_clean_images[im_idx])

        deg_image_pil = Image.open(deg_image_path).convert('L')
        clean_image_pil = Image.open(clean_image_path).convert('L')

        deg_image_np = np.array(deg_image_pil).astype(np.float32) / 255.0
        clean_image_np = np.array(clean_image_pil).astype(np.float32) / 255.0

        wat_batch_np, gt_batch_np = getPatches(deg_image_np,
                                               clean_image_np,
                                               mystride=128 + 64)

        all_wat_patches.append(wat_batch_np)
        all_gt_patches.append(gt_batch_np)

    all_wat_patches_tensor = torch.cat(all_wat_patches, dim=0).to(device)
    all_gt_patches_tensor = torch.cat(all_gt_patches, dim=0).to(device)

    dataset = PatchDataset(all_wat_patches_tensor, all_gt_patches_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    log_dir = "logs/gan_training_pytorch"
    discriminator_log_dir = os.path.join(log_dir, "discriminator")
    generator_log_dir = os.path.join(log_dir, "generator")

    writer_d = SummaryWriter(discriminator_log_dir)
    writer_g = SummaryWriter(generator_log_dir)

    global_step = 0
    last_g_loss_epoch = float('inf')

    for epoch in range(ep_start, epochs + 1):
        print(f"\nEpoch: {epoch}")
        generator.train()
        discriminator.train()

        g_loss_epoch = 0

        for b, (b_wat_batch, b_gt_batch) in enumerate(
                tqdm(dataloader)
        ):
            b_wat_batch = b_wat_batch.to(device)
            b_gt_batch = b_gt_batch.to(device)

            optimizer_D.zero_grad()

            generated_images = generator(b_wat_batch).detach()

            real_output_d = discriminator(b_gt_batch, b_wat_batch)

            real_labels = torch.ones_like(real_output_d, device=device)
            d_loss_real = criterion_discriminator(real_output_d,
                                                  real_labels)

            fake_output_d = discriminator(generated_images, b_wat_batch)
            fake_labels = torch.zeros_like(fake_output_d, device=device)
            d_loss_fake = criterion_discriminator(fake_output_d,
                                                  fake_labels)

            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            optimizer_D.step()

            d_acc_real = ((real_output_d >= 0.5).float().mean() * 100).item()
            d_acc_fake = ((fake_output_d < 0.5).float().mean() * 100).item()
            d_acc = 0.5 * (d_acc_real + d_acc_fake)

            writer_d.add_scalar('Loss/d_loss_total',
                                d_loss.item(),
                                global_step)
            writer_d.add_scalar('Loss/d_loss_real',
                                d_loss_real.item(),
                                global_step)
            writer_d.add_scalar('Loss/d_loss_fake',
                                d_loss_fake.item(),
                                global_step)
            writer_d.add_scalar('Accuracy/d_accuracy',
                                d_acc,
                                global_step)

            optimizer_G.zero_grad()

            generated_images = generator(b_wat_batch)
            gen_output_d = discriminator(generated_images, b_wat_batch)
            gen_labels = torch.ones_like(gen_output_d, device=device)

            g_loss_gan = criterion_generator_gan(gen_output_d, gen_labels)

            g_loss_pixel = criterion_generator_pixel(generated_images,
                                                     b_gt_batch)

            g_loss = g_loss_gan + 100 * g_loss_pixel
            g_loss.backward()
            optimizer_G.step()

            writer_g.add_scalar('Loss/g_total_loss',
                                g_loss.item(),
                                global_step)
            writer_g.add_scalar('Loss/g_gan_loss',
                                g_loss_gan.item(),
                                global_step)
            writer_g.add_scalar('Loss/g_pixel_loss',
                                g_loss_pixel.item(),
                                global_step)

            g_acc_disc_output = (
                    (gen_output_d >= 0.5).float().mean() * 100
            ).item()
            writer_g.add_scalar('Accuracy/g_disc_output_accuracy',
                                g_acc_disc_output,
                                global_step)

            global_step += 1

            if b % 10 == 0:
                print(
                    f"Batch {b}/{len(dataloader)} | "
                    f"D Loss: {d_loss.item():.4f} (Acc: {d_acc:.2f}%) | "
                    f"G Loss: {g_loss.item():.4f} "
                    f"(GAN Loss: {g_loss_gan.item():.4f}, "
                    f"Pixel Loss: {g_loss_pixel.item():.4f}, "
                    f"Disc Acc: {g_acc_disc_output:.2f}%)"
                )

            g_loss_epoch += g_loss.item()

        g_loss_epoch /= len(dataloader)

        print(f"Epoch {epoch} finished. Average G Loss: {g_loss_epoch:.4f}")

        if g_loss_epoch < last_g_loss_epoch:
            epoch_weights_path = f'trained_weights_pytorch/epoch_{epoch}'
            os.makedirs(epoch_weights_path, exist_ok=True)

            torch.save(discriminator.state_dict(),
                       os.path.join(epoch_weights_path, 'discriminator.pth'))
            torch.save(generator.state_dict(),
                       os.path.join(epoch_weights_path, 'generator.pth'))
            print(
                f"Saved models for epoch {epoch} as G loss improved from "
                f"{last_g_loss_epoch:.4f} to {g_loss_epoch:.4f}"
            )
            last_g_loss_epoch = g_loss_epoch

        # if (epoch == 1 or epoch % 2 == 0):
        #     print("Running evaluation...")
        #     evaluate(generator, epoch)

    writer_d.close()
    writer_g.close()


def predic(generator, epoch):
    generator.eval()  # Set generator to evaluation mode
    output_dir = 'Results_pytorch/epoch' + str(epoch)
    os.makedirs(output_dir, exist_ok=True)

    # Assumes CLEAN/VALIDATION/DATA contains the images to predict on
    validation_images = [f for f in os.listdir(VALIDATION_DATA_PATH) if
                         f.endswith(".png")]
    validation_images.sort()

    with torch.no_grad():  # Disable gradient calculations during prediction
        for i, img_name in enumerate(tqdm(validation_images)):
            watermarked_image_path = os.path.join(VALIDATION_DATA_PATH,
                                                  img_name)

            test_image_pil = Image.open(watermarked_image_path).convert('L')
            test_image_np = np.array(test_image_pil).astype(np.float32) / 255.0

            h_orig, w_orig = test_image_np.shape

            # Pad the image to be divisible by 256 for consistent patching
            h_pad = ((
                                 h_orig // 256) + 1) * 256 if h_orig % 256 != 0 else h_orig
            w_pad = ((
                                 w_orig // 256) + 1) * 256 if w_orig % 256 != 0 else w_orig

            test_padding_np = np.zeros((h_pad, w_pad), dtype=np.float32)
            test_padding_np[:h_orig, :w_orig] = test_image_np

            test_image_tensor = torch.from_numpy(test_padding_np).unsqueeze(
                0).unsqueeze(0).to(device)  # Add batch and channel dims

            # Split into patches
            test_image_patches = split2(test_image_tensor, 1, h_pad, w_pad)

            predicted_list = []
            for l in range(test_image_patches.shape[0]):
                patch_input = test_image_patches[l].unsqueeze(0).to(
                    device)  # Add batch dim for single patch
                predicted_patch = generator(patch_input).squeeze(
                    0).cpu().numpy()  # Remove batch dim, move to CPU, to numpy
                predicted_list.append(predicted_patch)

            predicted_image_tensor = torch.from_numpy(
                np.array(predicted_list))  # Stack patches
            # Reshape to (N, C, H_patch, W_patch)
            predicted_image_tensor = predicted_image_tensor.view(-1, 1, 256,
                                                                 256)

            # Merge patches back to image
            predicted_image_full = merge_image2(predicted_image_tensor, h_orig,
                                                w_orig).numpy()

            # Post-processing
            predicted_image_full = (predicted_image_full * 255.0).astype(
                np.uint8)

            imageio.imwrite(os.path.join(output_dir, f'predicted{i + 1}.png'),
                            predicted_image_full)

    print(f"Prediction for epoch {epoch} complete.")


# def evaluate(generator, epoch):
#     predic(generator, epoch)  # Generate predictions first
#     avg_psnr = 0
#     num_evaluated_images = 0
#
#     prediction_results_path = 'Results_pytorch/epoch' + str(epoch)
#
#     # Assumes CLEAN/VALIDATION/GT contains the ground truth images
#     ground_truth_images = [f for f in os.listdir(VALIDATION_CLEAN_PATH) if
#                            f.endswith(".png")]
#     ground_truth_images.sort()
#
#     for i, gt_img_name in enumerate(tqdm(ground_truth_images)):
#         gt_image_path = os.path.join(VALIDATION_CLEAN_PATH, gt_img_name)
#         predicted_image_path = os.path.join(prediction_results_path,
#                                             f'predicted{i + 1}.png')  # Assuming predic saves as predicted1.png, predicted2.png etc.
#
#         if not os.path.exists(predicted_image_path):
#             print(
#                 f"Warning: Predicted image {predicted_image_path} not found. Skipping PSNR for this image.")
#             continue
#
#         test_image = plt.imread(
#             gt_image_path)  # Assumes plt.imread loads as [0,1] float or [0,255] uint8
#         predicted_image = plt.imread(predicted_image_path)
#
#         # Ensure consistent data type and range for PSNR calculation
#         # If plt.imread loads as uint8, convert to float and normalize to [0,1] or keep [0,255] and set max_val
#         if test_image.dtype == np.uint8:
#             test_image = test_image.astype(np.float32) / 255.0
#         if predicted_image.dtype == np.uint8:
#             predicted_image = predicted_image.astype(np.float32) / 255.0
#
#         current_psnr = psnr(test_image, predicted_image,
#                             max_val=1.0)  # max_val depends on input range
#         avg_psnr += current_psnr
#         num_evaluated_images += 1
#
#     if num_evaluated_images > 0:
#         avg_psnr /= num_evaluated_images
#         print(f'PSNR for epoch {epoch} = {avg_psnr:.4f}')
#     else:
#         print("No images were evaluated for PSNR.")


if __name__ == "__main__":
    epo = 1

    generator = Generator(input_channels=1, output_channels=1,
                          biggest_layer=1024)
    discriminator = Discriminator(input_channels=1)

    train_gan(generator, discriminator, ep_start=epo, epochs=100, batch_size=1)