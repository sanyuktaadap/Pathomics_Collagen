from extract_patches import patch_extraction
from epithelium_stroma_segmentation_no_model import get_epithelium_mask
from patchLevelFeatures import extract_patch_level_features
from patientLevelFeatures import extract_patient_level_features
from create_tissue_mask import generate_patch_fg_mask

import argparse
# import torch
import glob
import os
import cv2
from tqdm import tqdm

# 1. Extract patches from WSIs
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--slides_folder', help='folder containing WSIs', default='data/images/')
parser.add_argument('-p', '--patches_folder', help='Folder path to save extracted patches', default='data/patches/')
parser.add_argument('-i', '--bg_intensity_thresh', help='Background intensity threshold', default=235)
parser.add_argument('-b', '--bg_mask_folder', help='Folder path to save fg/bg masks', default='data/patches/')
parser.add_argument('-t', '--patch_tile_size', help='Tile size for extracted patches', default=3000)
parser.add_argument('-e', '--epi_mask_folder', help='Folder path to save epithelium segmentation masks', default='data/masks/')
# parser.add_argument('-m', '--model_path', help='Path to segmentation model (.pth file)', default='code/unet/epi_seg_unet.pth')
parser.add_argument('-w', '--win_sizes', help='Window Sizes to convole over the image', default=[60, 65, 70])
parser.add_argument('--out_patch_feat_folder', help='Folder path to save output patch level features', default='results/patch_features')
parser.add_argument('--out_patient_feat_folder', help='Folder path to save output patient level features', default='results/patient_features')
args = parser.parse_args()

slides_folder = args.slides_folder
patches_folder = args.patches_folder
bg_intensity_thresh = args.bg_intensity_thresh
bg_mask_folder = args.bg_mask_folder
patch_tile_size = args.patch_tile_size
epi_mask_folder = args.epi_mask_folder
# model_path = args.model_path
win_sizes = args.win_sizes
out_patch_feat_folder = args.out_patch_feat_folder
out_patient_feat_folder = args.out_patient_feat_folder

if __name__ == "__main__":

    # 1. Extract patches
    os.makedirs(patches_folder, exist_ok=True)
    files = glob.glob(slides_folder + "*")
    for file in files:
        print(file)
        patch_extraction(slides_folder, patches_folder, patch_tile_size)

    # 2. Segment foreground/background from patches
    os.makedirs(bg_mask_folder, exist_ok=True)
    # device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    # unet = torch.load(model_path, map_location=device)
    # unet.eval()

    patches = glob.glob(os.path.join(patches_folder, "*.png"))
    model_input_size = 750
    for patch in tqdm(patches):
        filename = os.path.basename(patch)

        bg_path = os.path.join(bg_mask_folder, filename)
        bg_mask = generate_patch_fg_mask(patch,
                            bg_path,
                            patch_size=patch_tile_size,
                            stride=patch_tile_size,
                            intensity_thresh=bg_intensity_thresh)
        cv2.imwrite(bg_path, bg_mask)

    # 3. Segment Epithelum from patches
        patch_img = cv2.imread(patch)
        bg_img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
        epi_mask = get_epithelium_mask(patch_img, bg_img)
        os.makedirs(epi_mask_folder, exist_ok=True)
        cv2.imwrite(os.path.join(epi_mask_folder, filename), epi_mask)

        # mask = get_epithelium_mask(
        #     model=unet,
        #     input_path=patch,
        #     model_input_size=model_input_size,
        #     device=device
        # )

        # save_patch_epithelium_stroma_mask(
        #     mask=mask,
        #     output_path=os.path.join(patch_mask_folder, filename),
        #     patch_size=patch_tile_size
        # )

    print(f"Epithelium Segmentation Done! Masks saved at {epi_mask_folder}")

    # 3. Extract patch level features
    os.makedirs(out_patch_feat_folder, exist_ok=True)
    extract_patch_level_features(
        patches_folder=patches_folder,
        epi_mask_folder=epi_mask_folder,
        bg_mask_folder=bg_mask_folder,
        win_sizes=win_sizes,
        output_feat_folder=out_patch_feat_folder
    )

    # 4. Extract patient level features
    os.makedirs(out_patient_feat_folder, exist_ok=True)
    slides = glob.glob(slides_folder + "*")
    extract_patient_level_features(
        slides=slides,
        patch_features=out_patch_feat_folder,
        output=out_patient_feat_folder
    )