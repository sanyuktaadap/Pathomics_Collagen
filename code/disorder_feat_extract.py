"""
Authors: Arpit Aggarwal, Himanshu Maurya, Haojia Li
File: Dependency file for extracting collagen features (standard file, no changes needed!)
"""

# header files to load
import numpy as np
from scipy.special import comb
from haralick_no_img_v2 import haralick_no_img_v2
from itertools import combinations


# function
def contrast_entropy(orients, areas, orient_num, orient_cooccur_scheme):
    p_orient_occur = np.zeros((orient_num+1, orient_num+1))
    for pair1 in range(0, orient_num+1):  # Adjusted for Python's 0-indexing
        for pair2 in range(pair1, orient_num+1):  # Adjusted for Python's 0-indexing
            if np.any(orients == pair1) and np.any(orients == pair2):
                if pair1 != pair2:
                    if orient_cooccur_scheme == 1:
                        p_orient_occur[pair1, pair2] = np.sum(areas[orients == pair1]) * np.sum(areas[orients == pair2])
                    elif orient_cooccur_scheme == 2:
                        p_orient_occur[pair1, pair2] = np.sum(orients == pair1) * np.sum(orients == pair2)
                else:
                    iden_angle_num = np.sum(orients == pair1)
                    if iden_angle_num == 1:
                        p_orient_occur[pair1, pair2] = 0
                    else:
                        if orient_cooccur_scheme == 1:
                            indices = np.where(orients == pair1)[0]

                            # Generating all 2-element combinations of these indices
                            ind_permutation = np.array(list(combinations(indices, 2)))
                            p_orient_occur[pair1, pair2] = np.sum(areas[ind_permutation[:, 0]] * areas[ind_permutation[:, 1]])
                        elif orient_cooccur_scheme == 2:
                            p_orient_occur[pair1, pair2] = comb(iden_angle_num, 2, exact=True)

    # Normalize co-occurrence matrix
    orient_occur_matrix = p_orient_occur / np.sum(p_orient_occur)
    orient_occur_feats = haralick_no_img_v2(orient_occur_matrix)
    return orient_occur_feats