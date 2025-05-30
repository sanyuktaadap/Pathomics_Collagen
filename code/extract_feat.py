"""
File: Dependency file for extracting collagen features (standard file, no changes needed!)
"""

from compute_bifs import compute_bifs as compute_bifs
from disorder_feat_extract import contrast_entropy as disorder_feat_extract

import numpy as np
import cv2
from skimage import measure
import math

def extract_collagen_feats(patch,
                 collagen_mask,
                 win_sizes=[200, 250, 300, 350, 400, 450, 500, 550, 600],
                 ):

    # collagen centroid and orientation information extraction
    # Finds connected components (collagen blobs) in the binary collagen mask.
    # Each collagen blob is labeled with a unique number using cv2.drawContours().
    collagen_mask = collagen_mask.astype(np.uint16)
    collagen_mask_np_u8 = collagen_mask.copy().astype(np.uint8)
    # contours: a list of shapes/objects (boundaries of collagen blobs).
    contours, _ = cv2.findContours(collagen_mask_np_u8,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Draw (fill) each contour with a unique label into the original collagen_mask
    for i, contour in enumerate(contours):
        cv2.drawContours(collagen_mask, [contour], -1, color=(i+1), thickness=-1)

    # Extract geometric properties from each collagen region: center, angle and size
    properties = ('centroid', 'orientation', 'area')
    collogen_props = measure.regionprops_table(collagen_mask, properties=properties)
    colg_center = np.array([collogen_props['centroid-0'],
                            collogen_props['centroid-1']]).T
    colg_area = collogen_props['area']
    colg_orient = collogen_props['orientation']
    colg_orient = np.array([math.degrees(orient) for orient in colg_orient])
    # Convert orientation (in degrees) into bins of 10 degrees.
    # Shifted by 9 to make bins non-negative.
    colg_orient_bin = np.fix(colg_orient / 10) + 9

    features = []

    # Extract features from local neighborhoods
    for win_size in win_sizes:
        step_size = win_size
        # Map: a 3D array used to store extracted collagen orientation features
        map = np.zeros((int((patch.shape[0] - win_size) / step_size) + 1,
                        int((patch.shape[1] - win_size) / step_size) + 1,
                        13))
        height, width = collagen_mask.shape
        # Find collagen centroids that fall within the window.
        for win_x in range(0, width - win_size + 1, step_size):
            for win_y in range(0, height - win_size + 1, step_size):
                in_window_indices = np.where(
                        (colg_center[:, 0] >= win_x) &
                        (colg_center[:, 0] < win_x + win_size) &
                        (colg_center[:, 1] >= win_y) &
                        (colg_center[:, 1] < win_y + win_size)
                )
                in_window_indices = in_window_indices[0]
                # If at least 2 regions are inside, compute haralick features
                if len(in_window_indices) >= 2:
                    inwin_colg_orient = colg_orient_bin[in_window_indices]
                    inwin_colg_area = colg_area[in_window_indices]
                    orient_occur_feats = disorder_feat_extract(inwin_colg_orient,
                                                                inwin_colg_area,
                                                                18,
                                                                1)
                    if orient_occur_feats is not None:
                        map[win_y // step_size, win_x // step_size, :] = np.array(list(orient_occur_feats.values()))

        # Mean and Max of the feature at index 4 (5th haralick feature: contrast_entropy) of the map
        mean = np.mean(map[:,:,4])
        max = np.max(map[:,:,4])

        features.append(mean)
        features.append(max)

    return features