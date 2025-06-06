"""
File: Main file to extract collagen features at patch level
"""

# Import necessary libraries and functions
from compute_bifs import compute_bifs as compute_bifs
import numpy as np
import cv2
from skimage import measure, morphology
from disorder_feat_extract import contrast_entropy as disorder_feat_extract
import math
import os
import glob
import argparse

# MAIN CODE

# Parameters for analysis
win_sizes = [100, 125, 150, 175, 200]  # List of window sizes for feature extraction
filter_scale = 3   # Filter scale for BIFS computation
feature_descriptor = 5  # Descriptor to identify collagen features
orient_num = 180 // 10  # Number of orientation bins

# Parse command-line arguments to specify input/output paths
parser = argparse.ArgumentParser()
parser.add_argument('--input_patch', help='Input patches', default='data/patches/')
parser.add_argument('--input_mask', help='Input masks', default='data/masks/')
parser.add_argument('--output_heatmaps_stroma_win', help='Output heatmaps for stromal areas', default='results/heatmaps_stroma/')
parser.add_argument('--output_heatmaps_peritumoral_win', help='Output heatmaps for peritumoral areas', default='results/heatmaps_peritumoral/')
args = parser.parse_args()

# Set folder paths from arguments
patches_folder = args.input_patch
mask_folder = args.input_mask
patches_files = glob.glob(patches_folder + "*png")  # Get all patch files

# Process each patch file
for file in patches_files:
    print(file)

    # Skip files without a corresponding mask
    mask_path = mask_folder + file.split("/")[-1]
    if not os.path.isfile(mask_path):
        continue

    features = []

    filename = file.split("/")[-1]
    if filename not in lis:
        continue

    # Read the patch and its corresponding mask
    mask = 255 - cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Invert the mask
    patch = cv2.imread(file)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # FIRST SET OF FEATURES: Extract features from entire stromal areas

    # Generate a collagen fiber mask using BIFS
    frag_thresh = filter_scale * 10  # Threshold to filter small objects
    bifs, jet = compute_bifs(patch, filter_scale, 0.015, 1.5)  # Compute BIFS
    collagen_mask = bifs == feature_descriptor  # Identify collagen regions
    collagen_mask = np.logical_and(collagen_mask, mask)  # Apply the tissue mask
    collagen_mask = morphology.remove_small_objects(collagen_mask.astype(bool), min_size=frag_thresh)  # Remove small regions

    # Extract collagen centroid and orientation information
    collagen_mask = collagen_mask.astype(np.uint16)
    collagen_mask_np_u8 = collagen_mask.copy().astype(np.uint8)
    contours, hierarchy = cv2.findContours(collagen_mask_np_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv2.drawContours(collagen_mask, [contour], -1, color=(i + 1), thickness=-1)  # Label connected regions

    properties = ('centroid', 'orientation', 'area')
    collogen_props = measure.regionprops_table(collagen_mask, properties=properties)
    colg_center = np.array([collogen_props['centroid-0'], collogen_props['centroid-1']]).T
    colg_area = collogen_props['area']
    colg_orient = collogen_props['orientation']
    colg_orient = np.array([math.degrees(orient) for orient in colg_orient])  # Convert orientation to degrees
    colg_orient_bin = np.fix(colg_orient / 10) + 9  # Bin the orientations

    # Extract features within sliding windows
    for win_size in win_sizes:
        step_size = win_size  # Set step size equal to the window size
        map = np.zeros((int((patch.shape[0] - win_size) / step_size) + 1, int((patch.shape[1] - win_size) / step_size) + 1, 13))
        height, width = collagen_mask.shape

        for win_x in range(0, width - win_size + 1, step_size):
            for win_y in range(0, height - win_size + 1, step_size):
                # Identify collagen fibers within the window
                in_window_indices = np.where(
                    (colg_center[:, 0] >= win_x) &
                    (colg_center[:, 0] < win_x + win_size) &
                    (colg_center[:, 1] >= win_y) &
                    (colg_center[:, 1] < win_y + win_size)
                )[0]

                # Extract features if the window contains sufficient fibers
                if len(in_window_indices) >= 2:
                    inwin_colg_orient = colg_orient_bin[in_window_indices]
                    inwin_colg_area = colg_area[in_window_indices]
                    orient_occur_feats = disorder_feat_extract(inwin_colg_orient, inwin_colg_area, orient_num, 1)

                    if orient_occur_feats is not None:
                        map[win_y // step_size, win_x // step_size, :] = np.array(list(orient_occur_feats.values()))

        # Save heatmaps for stromal areas
        heatmap_normalized = cv2.normalize(map[:, :, 4].T, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap_uint8 = np.uint8(heatmap_normalized)
        heatmap_colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colormap = cv2.resize(heatmap_colormap, (int(patch.shape[0]), int(patch.shape[1])))
        overlayed_image = cv2.addWeighted(patch, 0.2, heatmap_colormap, 0.8, 0)
        cv2.imwrite(args.output_heatmaps_stroma_win + file.split("/")[-1][:-4] + f"_stroma_{win_size}.png", overlayed_image)

    # SECOND SET OF FEATURES: Extract features from peritumoral areas

    # Dilate the mask to define peritumoral regions
    im_dilated = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    for _ in range(50):
        im_dilated = cv2.dilate(im_dilated, np.ones((5, 5), np.uint8), iterations=1)
    im_new = im_dilated
    for _ in range(150):
        im_new = cv2.dilate(im_new, np.ones((5, 5), np.uint8), iterations=1)

    # Generate a collagen fiber mask for peritumoral areas
    frag_thresh = filter_scale * 10
    bifs, jet = compute_bifs(patch, filter_scale, 0.015, 1.5)
    collagen_mask = bifs == feature_descriptor
    collagen_mask = np.logical_and(collagen_mask, im_new)
    collagen_mask = np.logical_and(collagen_mask, 255 - im_dilated)
    collagen_mask = morphology.remove_small_objects(collagen_mask.astype(bool), min_size=frag_thresh)

    # Extract collagen centroid and orientation information
    collagen_mask = collagen_mask.astype(np.uint16)
    collagen_mask_np_u8 = collagen_mask.copy().astype(np.uint8)
    contours, hierarchy = cv2.findContours(collagen_mask_np_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv2.drawContours(collagen_mask, [contour], -1, color=(i + 1), thickness=-1)

    properties = ('centroid', 'orientation', 'area')
    collogen_props = measure.regionprops_table(collagen_mask, properties=properties)
    colg_center = np.array([collogen_props['centroid-0'], collogen_props['centroid-1']]).T
    colg_area = collogen_props['area']
    colg_orient = collogen_props['orientation']
    colg_orient = np.array([math.degrees(orient) for orient in colg_orient])
    colg_orient_bin = np.fix(colg_orient / 10) + 9

    # Extract features within sliding windows for peritumoral areas
    for win_size in win_sizes:
        step_size = win_size
        map = np.zeros((int((patch.shape[0] - win_size) / step_size) + 1,
                        int((patch.shape[1] - win_size) / step_size) + 1,
                        13))
        height, width = collagen_mask.shape

        for win_x in range(0, width - win_size + 1, step_size):
            for win_y in range(0, height - win_size + 1, step_size):
                # Identify collagen fibers within the window
                in_window_indices = np.where(
                    (colg_center[:, 0] >= win_x) &
                    (colg_center[:, 0] < win_x + win_size) &
                    (colg_center[:, 1] >= win_y) &
                    (colg_center[:, 1] < win_y + win_size)
                )[0]

                # Extract features if the window contains sufficient fibers
                if len(in_window_indices) >= 2:
                    inwin_colg_orient = colg_orient_bin[in_window_indices]
                    inwin_colg_area = colg_area[in_window_indices]
                    orient_occur_feats = disorder_feat_extract(inwin_colg_orient, inwin_colg_area, orient_num, 1)

                    if orient_occur_feats is not None:
                        map[win_y // step_size, win_x // step_size, :] = np.array(list(orient_occur_feats.values()))

        # Save heatmaps for peritumoral areas
        heatmap_normalized = cv2.normalize(map[:, :, 4].T, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap_uint8 = np.uint8(heatmap_normalized)
        heatmap_colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colormap = cv2.resize(heatmap_colormap, (int(patch.shape[0]), int(patch.shape[1])))
        overlayed_image = cv2.addWeighted(patch, 0.2, heatmap_colormap, 0.8, 0)
        cv2.imwrite(args.output_heatmaps_peritumoral_win + file.split("/")[-1][:-4] + f"_peritumoral_{win_size}.png", overlayed_image)