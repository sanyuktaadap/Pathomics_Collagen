import os
import shutil
import numpy as np
import cv2
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage import io

def is_edge_patch(patch_path, patch_size=3000, intensity_thresh=220, patching="20x", remove_folder=None):
    """
    Detects whether a patch is likely an 'edge patch' (mostly background, tissue edge, or artifact)
    based on intensity and color distribution thresholds.

    Args:
        patch_path (str): Path to the patch image (e.g., .jpg or .png).
        patch_size (int): Size of patch (assumes square).
        intensity_thresh (int): Threshold to classify bright (white) pixels.
        patching (str): Magnification level ("2.5x", "5x", "10x", or "20x").
        remove_folder (str or None): If provided, moves the patch to this folder if detected as edge.

    Returns:
        bool: True if patch is edge/background and should be removed, False otherwise.
    """
    # --- Load patch ---
    patch_array = io.imread(patch_path)

    # --- Calculate percentage of bright (white) pixels ---
    count_white_pixels = np.where(
        (patch_array[:, :, 0] > intensity_thresh) &
        (patch_array[:, :, 1] > intensity_thresh) &
        (patch_array[:, :, 2] > intensity_thresh)
    )[0]
    percent_white = len(count_white_pixels) / (patch_size * patch_size)

    # --- H&E stain separation (HED space) ---
    ihc_hed = rgb2hed(patch_array)

    # Rescale Eosin channel to [0, 255] (nuclear stain intensity)
    e = rescale_intensity(
        ihc_hed[:, :, 1],
        out_range=(0, 255),
        in_range=(0, np.percentile(ihc_hed[:, :, 1], 99))
    )

    # HSV channels
    hsv = cv2.cvtColor(patch_array, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Dark pixels
    count_dark = np.where(
        (patch_array[:, :, 0] < 128) &
        (patch_array[:, :, 1] < 128) &
        (patch_array[:, :, 2] < 128)
    )[0]
    percent_dark = len(count_dark) / (patch_size * patch_size)

    percent_low_saturation = np.mean(s_channel < 128)
    percent_high_value = np.mean(v_channel > intensity_thresh)

    # --- Common rejection condition: mostly white or no nuclei ---
    if percent_white >= 0.5 or np.mean(e < 50) > 0.50 or np.mean(h_channel < 128) > 0.50:
        edge_patch = True
    else:
        edge_patch = False

    # # --- Magnification-specific thresholds ---
    # if patching == '2.5x':
    #     if percent_low_saturation > 0.98 or np.mean(s_channel) < 5 or percent_high_value > 0.98:
    #         edge_patch = True
    #     elif (percent_low_saturation > 0.95 and percent_high_value > 0.95) or percent_dark > 0.95 or percent_white > 0.75:
    #         edge_patch = True

    # elif patching == '5x':
    #     if percent_low_saturation > 0.98 or np.mean(s_channel) < 5 or percent_high_value > 0.98:
    #         edge_patch = True
    #     elif (percent_low_saturation > 0.9 and percent_high_value > 0.9) or percent_dark > 0.95 or percent_white > 0.7:
    #         edge_patch = True

    # elif patching == '10x':
    #     if percent_low_saturation > 0.975 or np.mean(s_channel) < 5 or percent_high_value > 0.975:
    #         edge_patch = True
    #     elif (percent_low_saturation > 0.88 and percent_high_value > 0.88) or percent_dark > 0.9 or percent_white > 0.6:
    #         edge_patch = True

    # elif patching == '20x':
    #     if percent_low_saturation > 0.9 or np.mean(s_channel) < 5 or percent_high_value > 0.9:
    #         edge_patch = True
    #     elif (percent_low_saturation > 0.85 and percent_high_value > 0.9) or percent_dark > 0.9 or percent_white > 0.6:
    #         edge_patch = True

    # --- Move file if needed ---
    if edge_patch and remove_folder:
        os.makedirs(remove_folder, exist_ok=True)
        shutil.move(patch_path, os.path.join(remove_folder, os.path.basename(patch_path)))

    return edge_patch

# Example usage:
patch_path = "data/hari_BC/test/patches/"
remove_folder = "data/hari_BC/test/removed_edge_patches/"
patch_files = [f for f in os.listdir(patch_path) if f.endswith('.png')]
for patch_file in patch_files:
    full_patch_path = os.path.join(patch_path, patch_file)
    is_edge = is_edge_patch(full_patch_path, patch_size=3000, intensity_thresh=220, patching="20x", remove_folder=remove_folder)
    if is_edge:
        print(f"Removed edge patch: {patch_file}")
    # else:
    #     print(f"Kept patch: {patch_file}")