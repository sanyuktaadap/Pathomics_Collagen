"""
File: Dependency file for extracting collagen features (standard file, no changes needed!)
"""

# header files to load
import numpy as np
import cv2
from dtg_filters_bank import dtg_filters_bank
from efficient_convolution import efficient_convolution


# function
def compute_bifs(im, sigma, epsilon, configuration=1):
    """
    Computes basic image features (BIFs) for an image using the scale-space theory

    Parameters:
    - im: Input image.
    - sigma: Filter scale (standard deviation for Gaussian kernel).
    - epsilon: Amount of the image classified as flat. Flatness threshold used in classification.
    - configuration: Configuration for computing eigenvalues (lambda and mu). Defaults to 1.

    Returns:
    - bifs: Image where each pixel is labeled with one of 7 basic image features.
    - jet: 6-channel array of derivative responses at each pixel.
           It captures how the intensity is changing in the neighborhood around that pixel.
    """

    # Ensure the input is a NumPy array
    if not isinstance(im, np.ndarray):
        raise ValueError('Image must be a numpy array')
    # Convert image to grayscale float64
    if im.dtype != np.float64:
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im.astype(np.float64) / 255.0

    # Define the derivative orders to compute:
    # [0,0] = original image, [1,0] = x-derivative, [0,1] = y-derivative,
    # [2,0], [1,1], [0,2] = second-order derivatives
    orders = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])

    # Initialize an array to hold the 6 derivative responses
    jet = np.zeros((6, im.shape[0], im.shape[1]), dtype=np.float64)

    # Generate Derivative of Gaussian (DtG) filters for the given sigma
    DtGfilters = dtg_filters_bank(sigma);
    DtGfilters = np.array(DtGfilters)

    # Apply each DtG filter to the image and store result in jet
    for i, order in enumerate(orders):
        # Apply separable filters and scale response based on sigma
        jet[i, :, :] = efficient_convolution(im,
                                             DtGfilters[i, 0, :],
                                             DtGfilters[i, 1, :]) * (sigma ** (sum(order)))
    jet = np.array(jet)

    # Compute λ and μ values from second-order derivatives
    if configuration == 1: # faster, simpler, good for general use.
        lambda_val = (jet[3] + jet[5])
        mu = np.sqrt(((jet[3] - jet[5]) ** 2) + (4 * jet[4] ** 2))
    else: # more accurate, interpretable, better for curvature-sensitive tasks.
        lambda_val = 0.5 * (jet[3] + jet[5])
        mu = np.sqrt(0.25 * ((jet[3] - jet[5]) ** 2) + jet[4] ** 2)

    # Initialize array to hold 7 classifier responses
    c = np.zeros((jet.shape[1], jet.shape[2], 7), dtype=np.float64)

    # Compute the 7 basic image feature classifier responses:
    # 0 = Flat, 1 = Slope (edges), 2 = Dark line, 3 = Bright line,
    # 4 = Dark blob, 5 = Bright blob, 6 = Saddle
    c[:, :, 0] = epsilon * jet[0]
    c[:, :, 1] = 2 * np.sqrt(jet[1] ** 2 + jet[2] ** 2) if configuration == 1 else np.sqrt(jet[1] ** 2 + jet[2] ** 2)
    c[:, :, 2] = lambda_val
    c[:, :, 3] = -lambda_val
    c[:, :, 4] = 2 ** (-1/2) * (mu + lambda_val)
    c[:, :, 5] = 2 ** (-1/2) * (mu - lambda_val)
    c[:, :, 6] = mu

    # Classify each pixel based on the highest response
    bifs = np.argmax(c, axis=2)

    return bifs, jet