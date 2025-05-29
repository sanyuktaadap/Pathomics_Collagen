"""
Authors: Arpit Aggarwal, Himanshu Maurya, Haojia Li
File: Dependency file for extracting collagen features (standard file, no changes needed!)
"""

# header files to load
import numpy as np
from scipy.ndimage import gaussian_filter


# function
def dtg_filters_bank(sigma, configuration=1):
    """
    Generates the Derivative-of-Gaussian (DtG) kernels for the computation of BIFs.

    Parameters:
    sigma (float): Standard deviation of the Gaussian kernel.
    configuration (int, optional): Determines the method to compute the kernels. Defaults to 1.

    Returns:
    list of numpy.ndarray: The DtG kernels.
    """
    x = np.arange(-5*sigma, 5*sigma)
    xSquared = x**2
    DtGkernels = []

    # legacy matlab equivalent
    if configuration == 1:
        baseKernel = np.exp(-xSquared / (2 * sigma**2))
        dKernel = [
            (1 / (np.sqrt(2) * sigma)) * baseKernel,
            -x * (1 / (np.sqrt(2) * sigma**3)) * baseKernel,
            (xSquared - sigma**2) * (1 / (np.sqrt(2) * sigma**5)) * baseKernel
        ]

        orders = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])

        for order in orders:
            DtGkernels.append([dKernel[order[1]], dKernel[order[0]]])
        float_formatter = "{:.4f}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})

    return DtGkernels