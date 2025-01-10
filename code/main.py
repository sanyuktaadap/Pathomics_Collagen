from extract_patches import patch_extraction
from epithelium_stroma_segmentation import get_patch_epithelium_stroma_mask, save_patch_epithelium_stroma_mask
from patchLevelFeatures import extract_patch_level_features
from patientLevelFeatures import extract_patient_level_features

import argparse
import torch
import glob
import os
from tqdm import tqdm

# 1. Extract patches from WSIs
parser = argparse.ArgumentParser()
parser.add_argument('--slides_folder', help='folder containing WSIs', default='data/images/')
parser.add_argument('--patches_folder', help='Folder path to save extracted patches', default='data/patches/')
parser.add_argument('--patch_tile_size', help='Tile size for extracted patches', default=3000)
parser.add_argument('--patch_mask_folder', help='Folder path to save epithelium/storma segmentation masks', default='data/masks/')
parser.add_argument('--model_path', help='Path to segmentation model (.pth file)', default='code/unet/epi_seg_unet.pth')
parser.add_argument('--out_patch_feat_folder', help='Folder path to save output patch level features', default='results/patch_features')
parser.add_argument('--out_patient_feat_folder', help='Folder path to save output patient level features', default='results/patient_features')
args = parser.parse_args()


if __name__ == "__main__":

    # 1. Extract patches
    os.makedirs(args.patches_folder, exist_ok=True)
    files = glob.glob(args.slides_folder + "*")
    for file in files:
        print(file)
        patch_extraction(args.slides_folder, args.patches_folder, args.patch_tile_size)

    # 2. Segment epithelum/stroma from patches
    os.makedirs(args.patch_mask_folder, exist_ok=True)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    unet = torch.load(args.model_path, map_location=device)
    unet.eval()

    patches = glob(args.patches_folder + "*")
    model_input_size = 750
    for patch in tqdm(patches):
        filename = patch.split("/")[-1]

        mask = get_patch_epithelium_stroma_mask(
            model=unet,
            input_path=patch,
            model_input_size=model_input_size,
            device=device
        )

        save_patch_epithelium_stroma_mask(
            mask=mask,
            output_path=os.path.join(args.patch_mask_folder, filename),
            patch_size=args.patch_tile_size
        )
    print(f"Epithelium/Stroma Segmentation Done! Masks saved at {args.patch_mask_folder}")

    # 3. Extract patch level features
    os.makedirs(args.out_patch_feat_folder, exist_ok=True)
    extract_patch_level_features(
        patches_folder=args.patches_folder,
        mask_folder=args.patch_mask_folder,
        output_feat_folder=args.out_patch_feat_folder
    )

    # 4. Extract patient level features
    os.makedirs(args.out_patient_feat_folder, exist_ok=True)
    slides_folder = args.slides_folder
    patch_features = args.out_patch_feat_folder
    slides = glob.glob(slides_folder + "*")
    extract_patient_level_features(
        slides,
        patch_features
    )