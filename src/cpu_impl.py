import numpy as np
from scipy.ndimage import convolve

def gaussian_kernel(size=7, sigma=None):
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if sigma is None:
        sigma = size / 6.0
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

def gaussian_blur_cpu(img, size=7, sigma=None):
    """Gaussian blur using SciPy on CPU."""
    kernel = gaussian_kernel(size, sigma)
    if img.ndim == 3:
        chan = [convolve(img[..., c], kernel, mode='reflect') for c in range(3)]
        return np.stack(chan, axis=2)
    else:
        return convolve(img, kernel, mode='reflect')
