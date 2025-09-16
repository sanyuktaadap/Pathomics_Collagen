import cv2
import numpy as np
from skimage import color
from skimage import morphology
from scipy.ndimage import binary_fill_holes
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_epi_stroma_fat_mask_hed(patch):
    # Convert BGR → RGB
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    # Convert to HED
    hed = color.rgb2hed(rgb)

    # --- Epithelium (from Hematoxylin channel) ---
    hematoxylin = hed[:, :, 0]
    _, epi_mask = cv2.threshold(
        (hematoxylin * 255).astype(np.uint8),
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Clean epithelium
    frag_thresh = 200
    epi_mask = morphology.remove_small_objects(epi_mask.astype(bool), min_size=frag_thresh)
    epi_mask = (epi_mask.astype(np.uint8)) * 255
    kernel = np.ones((2, 2), np.uint8)
    epi_mask = cv2.dilate(epi_mask, kernel, iterations=30)
    epi_mask = cv2.erode(epi_mask, kernel, iterations=30)
    epi_mask = binary_fill_holes(epi_mask.astype(bool)).astype(np.uint8) * 255

    # apply epi_mask on grayscale patch and make teh epithelium region black on the patch.
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray_patch[epi_mask.astype(bool)] = 0
    # threshold on white regions (intensity >= 230)
    _, fat_mask = cv2.threshold(gray_patch, 230, 255, cv2.THRESH_BINARY)
    # dialate + erode to get rid of fine lines in fatty tissue
    fat_mask = cv2.dilate(fat_mask, kernel, iterations=15)
    fat_mask = cv2.erode(fat_mask, kernel, iterations=15)

    return epi_mask, fat_mask


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
    os.makedirs(f"data/hari_BC/otsu/epithelium_mask/{cohort}/", exist_ok=True)
    os.makedirs(f"data/hari_BC/otsu/fat_mask/{cohort}/", exist_ok=True)

    data_path = f"data/hari_BC/patches/{cohort}/"

    paths = glob.glob(os.path.join(data_path, "*.png"))
    print(paths)

    for path in tqdm(paths):
        patch = cv2.imread(path)
        epi_mask, fat_mask = get_epi_stroma_fat_mask_hed(patch)
        filename = os.path.basename(path)
        cv2.imwrite(f"data/hari_BC/otsu/epithelium_mask/{cohort}/{filename}", epi_mask)
        cv2.imwrite(f"data/hari_BC/otsu/fat_mask/{cohort}/{filename}", fat_mask)

# data_path = "data/hari_BC/test/patches"

# os.makedirs(f"data/hari_BC/test/labeled_mask/", exist_ok=True)
# os.makedirs(f"data/hari_BC/test/epi_mask/", exist_ok=True)
# os.makedirs(f"data/hari_BC/test/stroma_mask/", exist_ok=True)
# os.makedirs(f"data/hari_BC/test/fat_mask2/", exist_ok=True)
# os.makedirs(f"data/hari_BC/test/visualize/", exist_ok=True)

# paths = glob.glob(os.path.join(data_path, "*.png"))
# print(paths)

# for path in tqdm(paths):
#     patch = cv2.imread(path)
#     epi_mask, fat_mask = get_epi_stroma_fat_mask_hed(patch)
#     filename = os.path.basename(path)
#     # cv2.imwrite(f"data/hari_BC/test/labeled_mask/{filename}", labeled_mask)
#     cv2.imwrite(f"data/hari_BC/test/epi_mask/{filename}", epi_mask)
#     # cv2.imwrite(f"data/hari_BC/test/stroma_mask/{filename}", stroma_mask)
#     cv2.imwrite(f"data/hari_BC/test/fat_mask2/{filename}", fat_mask)

#     # patch_rgb, mask_reb = visualize_patch_with_masks(patch, labeled_mask)
#     # cv2.imwrite(f"data/hari_BC/test/visualize/{filename}", cv2.cvtColor(np.hstack((patch_rgb, mask_reb)), cv2.COLOR_RGB2BGR))