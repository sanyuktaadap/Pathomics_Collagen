"""
File: Dependency file for extracting collagen features (standard file, no changes needed!)
"""

# header files to load
import numpy as np


# function
def haralick_no_img_v2(SGLD):
    # Calculate statistics
    # Get the indices (pi, pj) of non-zero entries in the SGLD matrix and their values p
    pi, pj = np.nonzero(SGLD)
    p = SGLD[pi, pj]

    if len(p) <= 1:
        return None

    # Normalize the co-occurrence matrix to form a probability distribution and readjust the angles
    p = p / np.sum(p)
    pi -= 1
    pj -= 1

    # Marginals (How often each intensity appears at all, regardless of its neighbor)
    px_all = np.sum(SGLD, axis=1)
    pxi = np.nonzero(px_all)
    px = px_all[pxi]
    px = px / np.sum(px)
    pxi = pxi[0] - 1

    py_all = np.sum(SGLD, axis=0).T
    pyi = np.nonzero(py_all)
    py = py_all[pyi]
    py = py / np.sum(py)
    pyi = pyi[0] - 1

    # Calculate contrast features
    # Group similar contrast values and compute associated probabilities.
    all_contrast = np.abs(pi - pj)
    all_contrast[all_contrast > 9] = 18 - all_contrast[all_contrast > 9]

    sorted_indices = np.argsort(all_contrast)
    sorted_contrast = all_contrast[sorted_indices]

    p_sorted = p[sorted_indices]

    ind = np.concatenate([np.where(np.diff(sorted_contrast) != 0)[0], [len(all_contrast) - 1]])
    contrast = sorted_contrast[ind]
    # print(f"sorted_contrast: {sorted_contrast}")
    pcontrast = np.diff(np.concatenate(([0], np.cumsum(p_sorted)[ind])))

    contrast_energy = np.sum(contrast**2 * pcontrast)
    contrast_inverse_moment = np.sum((1 / (1 + contrast**2)) * pcontrast)
    contrast_ave = np.sum(contrast * pcontrast)
    contrast_var = np.sum((contrast - contrast_ave)**2 * pcontrast)
    contrast_entropy = -np.sum(pcontrast * np.log(pcontrast))

    # Calculate intensity features
    # Group and aggregate probabilities over sorted intensities.
    all_intensity = (pi + pj) / 2
    sorted_intensity = np.sort(all_intensity)
    sind = np.argsort(all_intensity)
    ind = np.concatenate([np.where(np.diff(sorted_intensity) != 0)[0], [len(all_intensity)-1]])

    # Extract the unique, sorted intensities
    intensity = sorted_intensity[ind]

    # Calculate the cumulative sum of probabilities at the sorted indices
    pintensity = np.cumsum(p[sind])

    # Calculate the difference in cumulative sum, prepending a 0 to align with MATLAB's behavior
    pintensity = np.diff(np.insert(pintensity[ind], 0, 0))

    intensity_ave = np.sum(intensity * pintensity)
    intensity_variance = np.sum((intensity - intensity_ave)**2 * pintensity)
    intensity_entropy = -np.sum(pintensity * np.log(pintensity))

    # Calculate probability features
    entropy = -np.sum(p * np.log(p))
    energy = np.sum(p**2)

    # Calculate correlation features
    # Calculate linear correlation between row and column intensities.
    mu_x = np.sum(pxi * px)
    sigma_x = np.sqrt(np.sum((pxi - mu_x)**2 * px))
    mu_y = np.sum(pyi * py)
    sigma_y = np.sqrt(np.sum((pyi - mu_y)**2 * py))

    if sigma_x != 0 and sigma_y != 0:
        correlation = np.sum((pi - mu_x) * (pj - mu_y) * p) / (sigma_x * sigma_y)
    else:
        correlation = 0

    px_grid, py_grid = np.meshgrid(px, py)
    log_px_grid, log_py_grid = np.meshgrid(np.log(px), np.log(py))

    h1 = -np.sum(p * np.log(np.where(px_all[pj] * py_all[pi] != 0, px_all[pj] * py_all[pi], 1)))
    h2 = -np.sum(px_grid.flatten() * py_grid.flatten() * (log_px_grid.flatten() + \
                 log_py_grid.flatten()))
    hx = -np.sum(px * np.log(px))
    hy = -np.sum(py * np.log(py))

    # Compute entropic differences and mutual dependencies using marginal distributions.
    information_measure1 = (entropy - h1) / max(hx, hy)
    information_measure2 = np.sqrt(1 - np.exp(-2 * (h2 - entropy)))

    # 13 Haralick Features
    feats = {
        'contrast_energy': contrast_energy, #Measures overall intensity contrast strength; higher for sharper intensity transitions.
        'contrast_inverse_moment': contrast_inverse_moment, #Weighs uniformity; higher when neighboring intensities are similar.
        'contrast_ave': contrast_ave, #Average of contrast values between pixel pairs.
        'contrast_var': contrast_var, #Variance of contrast values; reflects diversity in local intensity differences.
        'contrast_entropy': contrast_entropy, #Randomness or disorder in contrast distribution.
        'intensity_ave': intensity_ave, #Mean of the average intensities between pixel pairs.
        'intensity_variance': intensity_variance, #Spread of intensity averages; higher means greater texture variability.
        'intensity_entropy': intensity_entropy, #Unpredictability in average intensity values.
        'entropy': entropy, #Global randomness or complexity in the texture.
        'energy': energy, #Sum of squared probabilities; higher for homogeneous textures.
        'correlation': correlation, #Measures linear dependency between row and column intensities.
        'information_measure1': information_measure1, #Degree of mutual dependence between pixel pairs based on entropy difference.
        'information_measure2': information_measure2 #Alternative measure of dependence using exponential entropy differences.
    }
    return feats