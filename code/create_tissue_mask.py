import os
import numpy as np
import cv2
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage import io
from tqdm import tqdm
import os


def is_edge_patch_array(patch_array, patch_size=3000, intensity_thresh=220):
    """
    Check if patch is edge/background using color + intensity heuristics.
    Works directly on image arrays.
    """
    # --- Calculate percentage of bright (white) pixels ---
    count_white_pixels = np.where(
        (patch_array[:, :, 0] > intensity_thresh) &
        (patch_array[:, :, 1] > intensity_thresh) &
        (patch_array[:, :, 2] > intensity_thresh)
    )[0]
    percent_white = len(count_white_pixels) / (patch_size * patch_size)

    # --- H&E stain separation (HED space) ---
    ihc_hed = rgb2hed(patch_array)

    # Rescale Eosin channel to [0, 255]
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

    # --- Common rejection condition ---
    if percent_white >= 0.5 or np.mean(e < 50) > 0.50 or np.mean(h_channel < 128) > 0.50:
        return True
    else:
        return False


def generate_patch_fg_mask(image_path, mask_save_path, patch_size=100, stride=None, intensity_thresh=220):
    """
    Generates a binary mask for the entire image using patch-based classification.
    - White (255) = foreground/tissue
    - Black (0) = background/edge
    """
    if stride is None:
        stride = patch_size  # non-overlapping patches

    # --- Load image ---
    img = io.imread(image_path)
    H, W, _ = img.shape

    # --- Initialize mask ---
    mask = np.zeros((H, W), dtype=np.uint8)

    # --- Sliding window ---
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]

            is_edge = is_edge_patch_array(patch, patch_size=patch_size, intensity_thresh=intensity_thresh)

            if not is_edge:
                mask[y:y+patch_size, x:x+patch_size] = 255  # Foreground

    return mask
    # # --- Save mask ---
    # os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
    # cv2.imwrite(mask_save_path, mask)

    # return mask


# Example usage
import glob
cohorts = ["Black_cohort", "White_cohort"]
for cohort in cohorts:
    patches_path = f"data/hari_BC/patches/{cohort}"
    patches = glob.glob(os.path.join(patches_path, "*"))
    output_path = f"data/hari_BC/bg_mask/{cohort}"
    os.makedirs(output_path, exist_ok=True)
    for image_path in tqdm(patches):
        image_name = os.path.basename(image_path)
        print(image_name)
        mask_save_path = os.path.join(output_path, image_name)
        mask = generate_patch_fg_mask(image_path, mask_save_path, patch_size=3000, stride=3000, intensity_thresh=235)
        cv2.imwrite(mask_save_path, mask)