import os
import argparse
import matplotlib.pyplot as plt
import cv2
import re
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import cv2

def read_gt_image(folder_path):
    gt_images = []
    files = os.listdir(folder_path)
    files.sort()
    for filename in files:
        if filename.endswith(f'0006_gt_rgb.png'):
                # Read and append the image
                rgb_path = os.path.join(folder_path, filename)
                rgb_image = cv2.imread(rgb_path)
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                gt_images.append(rgb_image)

    return gt_images

def calculate_psnr(original_img, processed_img, max_pixel_value=255):
    mse = np.mean( ((original_img - processed_img) ** 2).flatten() )
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

import numpy as np
import cv2
import matplotlib.pyplot as plt

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

def plot_images(folder_path1, folder_path2, folder_path3):
    # Lists to store images and their corresponding PSNR values for each folder
    gt_images = read_gt_image(folder_path3)
    gt_image = gt_images[0]
    images1, psnrs1 = load_images_and_psnrs(folder_path1)
    images2, psnrs2 = load_images_and_psnrs(folder_path2)

    # Display only the first and last images
    indices = np.array([1])  # Display the first and last images

    images1 = images1[indices]
    psnrs1 = psnrs1[indices]

    images2 = images2[indices]
    psnrs2 = psnrs2[indices]

    target_height = 400  # Set the target height for all images

    zoom_loc = (335, 620)

    # Create a grid layout
    h, w, _ = gt_image.shape
    H, W, _ = images1[0].shape

    gs = GridSpec(1, 6, width_ratios=[W/H, 1, W/H, 1, W/H, 1], hspace=0.0, wspace=0.0)  # Adjust hspace as needed

    # Set the aspect ratio to be equal for all subplots
    fig = plt.figure(figsize=(15, 8))
    axs = [fig.add_subplot(gs[0, j], aspect='equal') for j in range(6)]

    axs[4].imshow(gt_image)
    # axs[4].set_title("Reference")
    axs[4].axis('off')

    #gt_zoomed_region = gt_image[200:300, 400:500, :]
    gt_zoomed_region = gt_image[zoom_loc[0]:(zoom_loc[0]+100), zoom_loc[1]:(zoom_loc[1]+100), :]
    axs[5].imshow(gt_zoomed_region)
    axs[5].axis('off')

    # Draw a red rectangle on the full-sized images
    rect_gt = Rectangle((zoom_loc[1],zoom_loc[0]), 100, 100, linewidth=1, edgecolor='red', facecolor='none')
    rect_full = Rectangle((0, 0), 100-1, 100-1, linewidth=1, edgecolor='red', facecolor='none')
    axs[5].add_patch(rect_full)
    axs[4].add_patch(rect_gt)

    for i, (images, psnrs) in enumerate([(images1, psnrs1), (images2, psnrs2)], start=0):
        axs[i * 2].imshow(images[0])
        # if i == 1:
        #     axs[i * 2].set_title(f"Hard Point Mining (ours)")
        # else:
        #     axs[i * 2].set_title(f"Random Sampling")
        axs[i * 2].axis('off')
        #axs[i * 2].text(100, 580, f"PSNR {psnrs[0]:.4f}")
        axs[i * 2].text(0.5,-0.1, f"PSNR {psnrs[0]:.2f}", size=12, ha="center", 
         transform=axs[i * 2].transAxes)

        for j, (image, psnr) in enumerate(zip(images, psnrs), start=1):
            h, w, _ = image.shape
            #zoomed_region = image[200:300, 400:500, :]
            zoomed_region = image[zoom_loc[0]:(zoom_loc[0]+100), zoom_loc[1]:(zoom_loc[1]+100), :]
            axs[i * 2 + j].imshow(zoomed_region)
            axs[i * 2 + j].axis('off')
            #axs[i * 2 + j].text(10, 110, f"PSNR {(calculate_psnr(zoomed_region, gt_zoomed_region)):.4f}")
            #axs[i * 2 + j].text(0.5,-0.1, f"PSNR {(calculate_psnr(zoomed_region, gt_zoomed_region)):.2f}", size=12, ha="center", 
            #    transform=axs[i * 2 + j].transAxes)
            rect_full = Rectangle((0, 0), 100-1, 100-1, linewidth=1, edgecolor='red', facecolor='none')
            axs[i * 2 + j].add_patch(rect_full)

    plt.tight_layout()
    plt.show()



# Assume read_gt_image, load_images_and_psnrs, calculate_psnr functions are defined






def load_images_and_psnrs(folder_path):
    # Lists to store images and their corresponding PSNR values
    images = []
    psnrs = []
    gt_images = []

    files = os.listdir(folder_path)
    files.sort()

    # Loop through all files in the folder
    for filename in files:
        match = re.match(rf'ngp_ep(\d+)_0006_error_(\d+\.\d+).png', filename)

        if match:
            psnr = float(match[2])
            psnrs.append(psnr)

        if filename.endswith(f'0006_gt_rgb.png'):
            # Read and append the image
            rgb_path = os.path.join(folder_path, filename)
            rgb_image = cv2.imread(rgb_path)
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            gt_images.append(rgb_image)


        if filename.endswith(f'0006_rgb.png'):
            # Read and append the image
            rgb_path = os.path.join(folder_path, filename)
            rgb_image = cv2.imread(rgb_path)
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            images.append(rgb_image)

    return np.array(images), np.array(psnrs) # , np.array(gt_images)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot RGB images with PSNR values from two folders.')
    parser.add_argument('folder_path1', type=str, help='Path to the first folder containing images')
    parser.add_argument('folder_path2', type=str, help='Path to the second folder containing images')
    parser.add_argument('gt_imgs', type=str, help='Path to the folder containing groundtruth images')
    args = parser.parse_args()

    # Call the function with the provided folder paths
    plot_images(args.folder_path1, args.folder_path2, args.gt_imgs)
