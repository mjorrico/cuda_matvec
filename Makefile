all: matvecmul matvecmul_transpose matmatmul

matvecmul: matvecmul.cu
	nvcc -arch=sm_75 -O3 -lineinfo matvecmul.cu -o matvecmul -lcublas -lcurand

matvecmul_transpose: matvecmul_transpose.cu
	nvcc -arch=sm_75 -O3 -lineinfo matvecmul_transpose.cu -o matvecmul_transpose -lcublas -lcurand

matmatmul: matmatmul.cu
	nvcc -arch=sm_75 -O3 -lineinfo matmatmul.cu -o matmatmul -lcublas -lcurand

clean:
	rm matvecmul matvecmul_transpose matmatmul