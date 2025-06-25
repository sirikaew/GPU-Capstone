.PHONY: run-gpu run-cpu

run-gpu:
	python src/main.py --input sample.jpg --output blurred_gpu.jpg --mode gpu --kernel-size 11

run-cpu:
	python src/main.py --input sample.jpg --output blurred_cpu.jpg --mode cpu --kernel-size 11
