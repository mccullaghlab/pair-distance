
CUDA_FLAGS = -Wno-deprecated-gpu-targets -lcurand -use_fast_math
GNU_FLAGS = -I/usr/local/cuda/include -O3 -lcudart

CXX = g++
CUDA = nvcc

isspa: compute_pair_dist_mat.cpp pair_dist_cuda.cu pair_dist_cuda.h atom_class.cpp atom_class.h config_class.h config_class.cpp 
	$(CXX) $(GNU_FLAGS) -c compute_pair_dist_mat.cpp atom_class.cpp config_class.cpp
	$(CUDA) $(CUDA_FLAGS) -c pair_dist_cuda.cu
	$(CUDA) $(CUDA_FLAGS) compute_pair_dist_mat.o atom_class.o config_class.o pair_dist_cuda.o -o compute_pair_dist.x 


