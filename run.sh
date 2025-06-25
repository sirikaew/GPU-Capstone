#!/usr/bin/env bash
python src/main.py --input sample.jpg --output blurred_gpu.jpg --mode gpu --kernel-size 11
python src/main.py --input sample.jpg --output blurred_cpu.jpg --mode cpu --kernel-size 11
