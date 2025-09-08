import cv2
import numpy as np
from skimage import color
from skimage import morphology
from scipy.ndimage import binary_fill_holes
import os
import glob
from tqdm import tqdm

def get_epi_stroma_mask_hed(patch):
    # Convert BGR → RGB for skimage
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    # Convert to HED space
    hed = color.rgb2hed(rgb)
    # Hematoxylin = channel 0 (epithelium)
    hematoxylin = hed[:, :, 0]

    # Otsu threshold on Hematoxylin channel
    _, epi_mask = cv2.threshold(
        (hematoxylin * 255).astype(np.uint8),
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Remove small fragments
    frag_thresh = 200  # minimum object size in pixels
    epi_mask = morphology.remove_small_objects(epi_mask.astype(bool), min_size=frag_thresh)
    epi_mask = (epi_mask.astype(np.uint8)) * 255

    # Morphological dilation to grow epithelium islands
    kernel = np.ones((2, 2), np.uint8)
    epi_mask = cv2.dilate(epi_mask, kernel, iterations=20)

    # Morphological erosion to revert dilation effect on boundaries
    epi_mask = cv2.erode(epi_mask, kernel, iterations=10)

    # Fill small holes inside epithelium blobs
    epi_mask = binary_fill_holes(epi_mask.astype(bool))
    epi_mask = (epi_mask.astype(np.uint8)) * 255

    # cv2.imwrite(f"final.png", epi_mask)

    return epi_mask


cohorts = ["Black_cohort", "White_cohort"]
for cohort in cohorts:
    os.makedirs(f"data/hari_BC/otsu_masks/{cohort}/", exist_ok=True)

    data_path = f"data/hari_BC/patches/{cohort}/"

    paths = glob.glob(os.path.join(data_path, "*.png"))
    print(paths)

    for path in tqdm(paths):
        patch = cv2.imread(path)
        epi_mask = get_epi_stroma_mask_hed(patch)
        filename = os.path.basename(path)
        cv2.imwrite(f"data/hari_BC/otsu_masks/{cohort}/{filename}", epi_mask)