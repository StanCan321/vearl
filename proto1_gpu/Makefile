NVCC ?= nvcc
# A10 = Ampere GA102 = sm_86
ARCH ?= -gencode arch=compute_86,code=sm_86

bench: bench.cu
	$(NVCC) -O3 -std=c++17 $(ARCH) -o $@ $< -lcublas

clean:
	rm -f bench
