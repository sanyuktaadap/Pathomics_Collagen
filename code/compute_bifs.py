"""
Authors: Arpit Aggarwal, Himanshu Maurya, Haojia Li
File: Dependency file for extracting collagen features (standard file, no changes needed!)
"""

# header files to load
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from dtg_filters_bank import dtg_filters_bank
from efficient_convolution import efficient_convolution


# function
def compute_bifs(im, sigma, epsilon, configuration=1):
    """
    Computes basic image features (BIFs) for an image.

    Parameters:
    - im: Input image.
    - sigma: Filter scale.
    - epsilon: Amount of the image classified as flat.
    - configuration: Configuration for computing lambda and mu. Defaults to 1.

    Returns:
    - bifs: Basic image features computed for each pixel.
    - jet: Derivative responses for the image.
    """
    if not isinstance(im, np.ndarray):
        raise ValueError('Image must be a numpy array')
    if im.dtype != np.float64:
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im.astype(np.float64) / 255.0

    # Derivative orders list
    orders = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])

    # Compute jets
    jet = np.zeros((6, im.shape[0], im.shape[1]), dtype=np.float64)

    DtGfilters = dtg_filters_bank(sigma);
    DtGfilters = np.array(DtGfilters)

    for i, order in enumerate(orders):
        jet[i, :, :] = efficient_convolution(im, DtGfilters[i,0,:],DtGfilters[i,1,:]) * (sigma ** (sum(order)))
    jet = np.array(jet)

    if configuration == 1:
        lambda_val = (jet[3] + jet[5])
        mu = np.sqrt(((jet[3] - jet[5]) ** 2) + (4 * jet[4] ** 2))
    else:
        lambda_val = 0.5 * (jet[3] + jet[5])
        mu = np.sqrt(0.25 * ((jet[3] - jet[5]) ** 2) + jet[4] ** 2)

    # Initialize classifiers array
    c = np.zeros((jet.shape[1], jet.shape[2], 7), dtype=np.float64)

    # Compute classifiers based on configuration
    c[:, :, 0] = epsilon * jet[0]
    c[:, :, 1] = 2 * np.sqrt(jet[1] ** 2 + jet[2] ** 2) if configuration == 1 else np.sqrt(jet[1] ** 2 + jet[2] ** 2)
    c[:, :, 2] = lambda_val
    c[:, :, 3] = -lambda_val
    c[:, :, 4] = 2 ** (-1/2) * (mu + lambda_val)
    c[:, :, 5] = 2 ** (-1/2) * (mu - lambda_val)
    c[:, :, 6] = mu

    # Assign each pixel to the largest classifier
    bifs = np.argmax(c, axis=2)
    return bifs, jet