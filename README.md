# GPU Gaussian Blur Capstone Project

This project demonstrates a simple yet illustrative **Gaussian blur** image filter accelerated with NVIDIA GPUs via **CuPy**.
It also provides a pure‑CPU fallback so it can be executed on machines without CUDA hardware.

> *Course context:* CUDA/GPU Programming Specialization – Capstone  
> *Author:* Sirikaew 

## Why Gaussian Blur?
* Convolution‑based blurs are common in computer vision.  
* They translate nicely into massively parallel GPU kernels.  
* Performance can be compared easily between CPU and GPU.

---

## Repository Layout
```
gpu_capstone_project/
├── README.md
├── requirements.txt
├── Makefile
├── run.sh
├── sample.jpg
├── logs/
│   └── sample_cpu_run.txt
└── src/
    ├── __init__.py
    ├── main.py
    ├── cpu_impl.py
    ├── gpu_impl.py
    └── utils.py
```

## Quick Start (GPU)

```bash
# 1. Create a fresh virtual env (optional but recommended)
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies *with* GPU support
pip install -r requirements.txt
# 👉 For GPU you need **CuPy** built against your CUDA version.
# Example for CUDA 11.8 on Linux:
#   pip install cupy-cuda118
# Other CUDA versions:               cupy-cuda11x
# ROCm users (AMD GPUs):             cupy-rocm-5xx
# Or see: https://docs.cupy.dev/en/stable/install.html

# 3. Run blur on the GPU
python src/main.py --input sample.jpg --output blurred_gpu.jpg --mode gpu --kernel-size 11
```

## Running on **CPU‑only** Hardware 🖥️

If your system has **no GPU** or you don't wish to install CuPy:

```bash
pip install -r requirements.txt           # will *not* install CuPy
python src/main.py --input sample.jpg --output blurred_cpu.jpg --mode cpu --kernel-size 11
```

The script **auto‑detects** CuPy at runtime. If the `--mode gpu` flag is supplied but CuPy is missing, it will transparently fall back to the CPU implementation and issue a warning.

## CLI Reference

```
usage: main.py [-h] --input INPUT [--output OUTPUT] [--mode {gpu,cpu,auto}] [--kernel-size KSIZE] [--sigma SIGMA]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         path to input image
  --output OUTPUT       path for result (default: blurred.jpg)
  --mode {gpu,cpu,auto} processing mode (default: auto → GPU if available)
  --kernel-size KSIZE   Gaussian kernel size (odd integer, default: 7)
  --sigma SIGMA         Gaussian σ (default: KSIZE/6)
```

## Proof of Execution

A sample run on a **CPU‑only** notebook VM is stored in `logs/sample_cpu_run.txt`.  
Example excerpt:

```
Mode: CPU   Kernel: 11×11   Elapsed: 0.127 s
```

Feel free to replace it with your own benchmarking logs or screenshots for presentation.

## Presentation

Record a 5‑10 min video summarising:

1. **Goal** – fast Gaussian blur & GPU vs CPU speedup  
2. **Design** – CuPy vs NumPy paths, tiling & shared memory (brief)  
3. **Results** – show timing table & blurred images  
4. **Lessons Learned / Next Steps**

Upload the video (YouTube unlisted, Google Drive, etc.) and link it in this README.

Happy hacking! 🎉
