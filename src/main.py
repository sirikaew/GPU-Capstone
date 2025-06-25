#!/usr/bin/env python3
"""Entry point for Gaussian blur project."""
import argparse, time, os, warnings
from utils import load_image, save_image

# Try GPU import
try:
    from gpu_impl import gaussian_blur_gpu
    HAS_GPU = True
except Exception as e:
    HAS_GPU = False

from cpu_impl import gaussian_blur_cpu

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='input image path')
    p.add_argument('--output', default='blurred.jpg', help='output image path')
    p.add_argument('--mode', choices=['gpu','cpu','auto'], default='auto',
                   help='processing mode (default: auto)')
    p.add_argument('--kernel-size', type=int, default=7, help='odd kernel size')
    p.add_argument('--sigma', type=float, default=None, help='Gaussian sigma (default size/6)')
    return p.parse_args()

def main():
    args = parse()
    img = load_image(args.input)
    mode = args.mode
    if mode == 'auto':
        mode = 'gpu' if HAS_GPU else 'cpu'

    start = time.time()
    if mode == 'gpu':
        if not HAS_GPU:
            warnings.warn('GPU path requested but CuPy unavailable – falling back to CPU.')
            mode = 'cpu'
            out = gaussian_blur_cpu(img, args.kernel_size, args.sigma)
        else:
            out = gaussian_blur_gpu(img, args.kernel_size, args.sigma)
    if mode == 'cpu':
        out = gaussian_blur_cpu(img, args.kernel_size, args.sigma)
    elapsed = time.time() - start

    print(f"Mode: {mode.upper():<4}  Kernel: {args.kernel_size}×{args.kernel_size}   Elapsed: {elapsed:.3f} s")
    save_image(out, args.output)

if __name__ == '__main__':
    main()
