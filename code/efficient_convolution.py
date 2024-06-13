"""
Authors: Arpit Aggarwal, Himanshu Maurya, Haojia Li
File: Dependency file for extracting collagen features (standard file, no changes needed!)
"""

# header files to load
from scipy.ndimage import convolve
import numpy as np


# function
def efficient_convolution(I, kx, ky):
    """
    Convolution of an image using two separable 1-D kernels.
    
    Parameters:
    I (numpy.ndarray): Input image.
    kx (numpy.ndarray): Kernel to be applied on the horizontal axis.
    ky (numpy.ndarray): Kernel to be applied on the vertical axis.
    
    Returns:
    numpy.ndarray: The convolved image.
    """
    # Convolve with kx. The mode 'reflect' replicates the edge values.
    # none of the modes are equivalent to 'same' in MATLAB ??
    J = convolve(I, kx[None, :], mode='reflect')
    # Convolve with ky. Need to transpose ky to match MATLAB's behavior.
    J = convolve(J, ky[:, None], mode='reflect')
    return J