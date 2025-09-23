import cv2
import numpy as np
from skimage import color
from skimage import morphology
from scipy.ndimage import binary_fill_holes
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_epithelium_mask(patch, fg_mask):
    """
    Generate epithelium mask from a patch using the hematoxylin channel,
    guided by a pre-computed foreground/background mask.

    Args:
        patch (np.array): BGR patch image.
        fg_mask (np.array): Binary mask of foreground (255 = tissue, 0 = background).

    Returns:
        epi_mask (np.array): Binary epithelium mask (255 = epithelium, 0 = stroma/background).
    """
    # Ensure masks are single channel binary
    fg_mask = (fg_mask > 0).astype(np.uint8)

    # Convert BGR → RGB → HED
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    hed = color.rgb2hed(rgb)
    hematoxylin = hed[:, :, 0]

    # Scale hematoxylin channel to [0, 255]
    hematoxylin_scaled = cv2.normalize(hematoxylin, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply foreground mask → only keep tissue region
    tissue_hema = cv2.bitwise_and(hematoxylin_scaled, hematoxylin_scaled, mask=fg_mask)

    # Otsu thresholding within tissue region
    _, epi_mask = cv2.threshold(tissue_hema, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Restrict result strictly to tissue area
    epi_mask = cv2.bitwise_and(epi_mask, epi_mask, mask=fg_mask)

    # Morphological cleaning
    epi_mask = morphology.remove_small_objects(epi_mask.astype(bool), min_size=200)
    epi_mask = (epi_mask.astype(np.uint8)) * 255
    kernel = np.ones((2, 2), np.uint8)
    epi_mask = cv2.dilate(epi_mask, kernel, iterations=60)
    epi_mask = cv2.erode(epi_mask, kernel, iterations=50)
    epi_mask = binary_fill_holes(epi_mask.astype(bool)).astype(np.uint8) * 255

    return epi_mask


def visualize_patch_with_masks(patch, labeled_mask):
    """Show patch and labeled mask side by side with colors."""
    # Define colormap: (R, G, B)
    colors = {
        0: (0, 0, 0),        # background = black
        1: (255, 0, 0),      # epithelium = red
        2: (0, 255, 0),      # stroma = green
        3: (0, 0, 255)       # fat = blue
    }

    # Create RGB mask
    mask_rgb = np.zeros((*labeled_mask.shape, 3), dtype=np.uint8)
    for k, col in colors.items():
        mask_rgb[labeled_mask == k] = col

    # Convert BGR patch to RGB for plotting
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    return patch_rgb, mask_rgb

cohorts = ["Black_cohort", "White_cohort"]
for cohort in cohorts:
    os.makedirs(f"data/hari_BC/otsu/epi_mask_no_bg/{cohort}/", exist_ok=True)

    data_path = f"data/hari_BC/patches/{cohort}/"
    fg_paths = f"data/hari_BC/bg_mask/{cohort}"

    paths = glob.glob(os.path.join(data_path, "*.png"))
    print(paths)

    for path in tqdm(paths):
        name = os.path.basename(path)
        fg_path = os.path.join(fg_paths, name)
        patch = cv2.imread(path)
        fg_mask = cv2.imread(fg_path, cv2.IMREAD_GRAYSCALE)
        epi_mask = get_epithelium_mask(patch, fg_mask)
        filename = os.path.basename(path)
        cv2.imwrite(f"data/hari_BC/otsu/epi_mask_no_bg/{cohort}/{filename}", epi_mask)