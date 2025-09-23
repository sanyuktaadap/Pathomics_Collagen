"""
File: Main file to extract collagen features at patch level
"""

# header files to load
from compute_bifs import compute_bifs as compute_bifs
import numpy as np
import cv2
from skimage import morphology
import os
import glob
import argparse
import csv
from extract_feat import extract_collagen_feats
from tqdm import tqdm

def extract_patch_level_features(patches_folder, epi_mask_folder, bg_mask_folder, win_sizes, output_feat_folder):
    filter_scale = 3
    feat = 5
    patches_files = glob.glob(os.path.join(patches_folder, "*png"))
    print(f"patches_folder: {patches_folder}")

    for file in tqdm(patches_files):
        file_name = file.split("/")[-1]

        if not os.path.isfile(os.path.join(epi_mask_folder, file_name)) or not os.path.isfile(os.path.join(bg_mask_folder, file_name)):
            continue

        features = []

        # read patch and mask
        patch = cv2.imread(file)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        epi_mask = cv2.imread(os.path.join(epi_mask_folder, file_name),
                            cv2.IMREAD_GRAYSCALE)
        epi_mask = 255 - epi_mask
        bg_mask = cv2.imread(os.path.join(bg_mask_folder, file_name),
                              cv2.IMREAD_GRAYSCALE)

        # FIRST SET OF FEATURES FROM ENTIRE STROMAL AREAS
        # extract collagen fiber mask
        frag_thresh = filter_scale * 10
        bifs, _ = compute_bifs(patch, filter_scale, 0.015, 1.5)
        collagen_mask = bifs == feat
        collagen_mask = np.logical_and(collagen_mask, epi_mask)
        collagen_mask = np.logical_and(collagen_mask, bg_mask)
        collagen_mask = morphology.remove_small_objects(collagen_mask.astype(bool),
                                                        min_size=frag_thresh)

        stroma_features = extract_collagen_feats(patch, collagen_mask, win_sizes)
        features.extend(stroma_features)

        # SECOND SET OF FEATURES FROM PERITUMORAL AREAS
        # get dilated mask
        im_dilated = cv2.imread(os.path.join(epi_mask_folder, file_name),
                                cv2.IMREAD_GRAYSCALE)
        for index1 in range(0, 15):
            im_dilated = cv2.dilate(im_dilated, np.ones((5, 5), np.uint8), iterations=1)
        im_new = im_dilated
        for index1 in range(0, 30):
            im_new = cv2.dilate(im_new, np.ones((5, 5), np.uint8), iterations=1)

        # extract collagen fiber mask
        frag_thresh = filter_scale * 10
        bifs, _ = compute_bifs(patch, filter_scale, 0.015, 1.5)
        collagen_mask = bifs == 5
        collagen_mask = np.logical_and(collagen_mask, im_new)
        collagen_mask = np.logical_and(collagen_mask, 255 - im_dilated)
        collagen_mask = np.logical_and(collagen_mask, epi_mask)  # Exclude epithelial regions
        collagen_mask = np.logical_and(collagen_mask, bg_mask)  # Exclude fat regions
        collagen_mask = morphology.remove_small_objects(collagen_mask.astype(bool),
                                                        min_size=frag_thresh)

        # collagen centroid and orientation information extraction
        peri_tumor_feats = extract_collagen_feats(patch, collagen_mask, win_sizes)
        features.extend(peri_tumor_feats)

        file_name = file_name.rsplit('.', 1)[0] + ".csv"
        with open(os.path.join(output_feat_folder, file_name), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(features)


if __name__ == "__main__":
    # MAIN CODE
    # win_sizes = [200, 250, 300, 350, 400, 450, 500, 550, 600]
    # filter_scale = 3
    # feature_descriptor = 5
    # orient_num = 18 #180 // 10

    # read patches and masks
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--input_patch', help='Input patches', default='data/patches/')
    parser.add_argument("-e", '--epi_mask', help='Input masks', default='data/masks/')
    parser.add_argument("-b", '--bg_mask', help='Input fat masks', default='data/bg_mask/')
    parser.add_argument("-o", '--output_feature', help='Output Features Folder', default='results/patch_features/')
    parser.add_argument('--win_sizes', help='Window Sizes to convole over the image', default=[60,65,70])
    args = parser.parse_args()

    patches_folder = args.input_patch
    epi_mask_folder = args.epi_mask
    bg_mask_folder = args.bg_mask
    output_feat_folder = args.output_feature
    win_sizes = args.win_sizes

    cohorts = os.listdir(patches_folder)
    for cohort in cohorts:
        cohort_patch_fold = os.path.join(patches_folder, cohort)
        cohort_epi_mask_fold = os.path.join(epi_mask_folder, cohort)
        cohort_bg_mask_fold = os.path.join(bg_mask_folder, cohort)
        os.makedirs(os.path.join(output_feat_folder, cohort), exist_ok=True)
        extract_patch_level_features(cohort_patch_fold, cohort_epi_mask_fold, cohort_bg_mask_fold, win_sizes, os.path.join(output_feat_folder, cohort))