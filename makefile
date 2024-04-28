all:
	nvcc -Xcompiler -fPIC -shared -o ant_solution.so ant_solution.cu

run:
	nvcc -Xcompiler -fPIC -shared -o ant_solution.so ant_solution.cu
	python3 aco.py

sorted:
	nvcc -Xcompiler -fPIC -shared -o ant_solution.so ant_solution.cu
	python3 aco.py | sort
