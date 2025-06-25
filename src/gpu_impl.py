"""GPU Gaussian blur using CuPy."""
try:
    import cupy as cp
    from cupyx.scipy.signal import convolve2d
except Exception as e:
    cp = None  # gracefully degrade

def _gaussian_kernel(size=7, sigma=None):
    if sigma is None:
        sigma = size / 6.0
    ax = cp.linspace(-(size // 2), size // 2, size)
    xx, yy = cp.meshgrid(ax, ax)
    kernel = cp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= cp.sum(kernel)
    return kernel.astype(cp.float32)

def gaussian_blur_gpu(img, size=7, sigma=None):
    if cp is None:
        raise RuntimeError("CuPy not available – cannot run GPU path.")
    img_gpu = cp.asarray(img, dtype=cp.float32)
    kernel = _gaussian_kernel(size, sigma)

    if img_gpu.ndim == 3:
        out_ch = [convolve2d(img_gpu[..., c], kernel, mode='same', boundary='symm')
                  for c in range(3)]
        out = cp.stack(out_ch, axis=2)
    else:
        out = convolve2d(img_gpu, kernel, mode='same', boundary='symm')
    return cp.asnumpy(out)
