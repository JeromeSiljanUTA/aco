all:
	nvcc -Xcompiler -fPIC -shared -o dist_sum.so cuda_module.cu

run:
	nvcc -Xcompiler -fPIC -shared -o dist_sum.so cuda_module.cu
	python main.py
