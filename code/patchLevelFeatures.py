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


# MAIN CODE
# win_sizes = [200, 250, 300, 350, 400, 450, 500, 550, 600]
# filter_scale = 3
# feature_descriptor = 5
# orient_num = 18 #180 // 10

# read patches and masks
parser = argparse.ArgumentParser()
parser.add_argument('--input_patch', help='Patches Folder', default='data/patches/')
parser.add_argument('--input_mask', help='Masks Folder', default='data/masks/')
parser.add_argument('--output_feature', help='Output Features Folder', default='results/patch_features/')
args = parser.parse_args()

patches_folder = args.input_patch
mask_folder = args.input_mask
output_feat_folder = args.output_feature

def extract_patch_level_features(patches_folder, mask_folder, output_feat_folder):
    patches_files = glob.glob(patches_folder+"*png")

    for file in patches_files:
        print(file)
        if os.path.isfile(os.path.join(mask_folder, file.split("/")[-1])) == 0:
            continue

        features = []

        # read patch and mask
        mask = 255 - cv2.imread(os.path.join(mask_folder, file.split("/")[-1]), cv2.IMREAD_GRAYSCALE)
        patch = cv2.imread(file)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        # FIRST SET OF FEATURES FROM ENTIRE STROMAL AREAS
        # extract collagen fiber mask
        frag_thresh = 3 * 10
        bifs, _ = compute_bifs(patch, 3, 0.015, 1.5)
        collagen_mask = bifs == 5
        collagen_mask = np.logical_and(collagen_mask, mask)
        collagen_mask = morphology.remove_small_objects(collagen_mask.astype(bool), min_size=frag_thresh)

        # collagen centroid and orientation information extraction
        win_sizes = [200, 250, 300, 350, 400, 450, 500, 550, 600]

        stroma_features = extract_collagen_feats(patch, collagen_mask, win_sizes)
        features.extend(stroma_features)

        # SECOND SET OF FEATURES FROM PERITUMORAL AREAS
        # get dilated mask
        im_dilated = cv2.imread(os.path.join(mask_folder, file.split("/")[-1]), cv2.IMREAD_GRAYSCALE)
        for index1 in range(0, 50):
            im_dilated = cv2.dilate(im_dilated, np.ones((5, 5), np.uint8), iterations=1)
        im_new = im_dilated
        for index1 in range(0, 150):
            im_new = cv2.dilate(im_new, np.ones((5, 5), np.uint8), iterations=1)

        # extract collagen fiber mask
        frag_thresh = 3 * 10
        bifs, _ = compute_bifs(patch, 3, 0.015, 1.5)
        collagen_mask = bifs == 5
        collagen_mask = np.logical_and(collagen_mask, im_new)
        collagen_mask = np.logical_and(collagen_mask, 255-im_dilated)
        collagen_mask = morphology.remove_small_objects(collagen_mask.astype(bool), min_size=frag_thresh)

        # collagen centroid and orientation information extraction
        peri_tumor_feats = extract_collagen_feats(patch, collagen_mask, win_sizes)
        features.extend(peri_tumor_feats)

        file_name = file.split("/")[-1]
        file_name = file_name.rsplit('.', 1)[0] + ".csv"
        with open(os.path.join(output_feat_folder, file_name), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(features)


if __name__ == "__main__":
    extract_patch_level_features(patches_folder, mask_folder, output_feat_folder)