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
    """
    Computes texture features from an orientation co-occurrence matrix.

    Inputs:
    orients : np.ndarray
        1D array of integer orientation labels assigned to image regions or patches.

    areas : np.ndarray
        1D array of region areas corresponding to each orientation label in `orients`.

    orient_num : int
        The maximum orientation label (e.g., if orientations range from 0 to 8, orient_num = 8).

    orient_cooccur_scheme : int
        Scheme to compute co-occurrence strength:
            1 - Area-weighted co-occurrence: Uses product of region areas.
            2 - Count-based co-occurrence: Uses frequency of orientation pairs.

    Returns:
    orient_occur_feats : dict
        Dictionary of Haralick-style texture features computed from the normalized
        orientation co-occurrence matrix. Includes features like contrast, entropy,
        correlation, energy, and information measures.

    Notes:
    - The orientation co-occurrence matrix is symmetric and represents how often
      orientation pairs co-occur across the image.
    - For identical orientation pairs (diagonal), the matrix is populated by computing
      pairwise region interactions (area products or combinations).
    - Normalized matrix is passed to `haralick_no_img_v2()` for texture feature extraction.
    """

    p_orient_occur = np.zeros((orient_num+1, orient_num+1))
    for pair1 in range(0, orient_num+1):  # Adjusted for Python's 0-indexing
        for pair2 in range(pair1, orient_num+1):  # Adjusted for Python's 0-indexing
            if np.any(orients == pair1) and np.any(orients == pair2):
                if pair1 != pair2:
                    if orient_cooccur_scheme == 1:
                        p_orient_occur[pair1, pair2] = np.sum(areas[orients == pair1]) * \
                                                       np.sum(areas[orients == pair2])
                    elif orient_cooccur_scheme == 2:
                        p_orient_occur[pair1, pair2] = np.sum(orients == pair1) * \
                                                       np.sum(orients == pair2)
                else:
                    iden_angle_num = np.sum(orients == pair1)
                    if iden_angle_num == 1:
                        p_orient_occur[pair1, pair2] = 0
                    else:
                        if orient_cooccur_scheme == 1:
                            indices = np.where(orients == pair1)[0]

                            # Generating all 2-element combinations of these indices
                            ind_permutation = np.array(list(combinations(indices, 2)))
                            p_orient_occur[pair1, pair2] = np.sum(areas[ind_permutation[:, 0]] * \
                                                                  areas[ind_permutation[:, 1]])
                        elif orient_cooccur_scheme == 2:
                            p_orient_occur[pair1, pair2] = comb(iden_angle_num, 2, exact=True)

    # Normalize co-occurrence matrix
    orient_occur_matrix = p_orient_occur / np.sum(p_orient_occur)
    orient_occur_feats = haralick_no_img_v2(orient_occur_matrix)
    return orient_occur_feats