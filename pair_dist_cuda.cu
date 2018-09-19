
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "pair_dist_cuda.h"

#define nDim 3
//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

//__global__ void init_rand(unsigned int long seed, curandState_t* states){
//	curand_init(seed,blockIdx.x,0,&states);
//}
// CUDA Kernels

__global__ void pair_dist_kernel(float *xyz, float *pairDist, int nAtoms, unsigned int *blockID) {
	extern __shared__ float xyz_shared[];
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom1;
	int atom2;
	int it;    // atom type of atom of interest
	int jt;    // atom type of other atom
	float temp, dist2;	
	int i, k;
	int count;
	int start;
	float r[3];
//	__shared__ float xyz_shared[nAtoms*nDim];

	if (index < nAtoms)
	{
		blockID[index] = blockIdx.x;
		atom1 = index;
		// copy atom1 position to local shared memory
		for (k=0;k<nDim;k++) {
			xyz_shared[atom1*nDim+k] = xyz[atom1*nDim+k];	
		}
		__syncthreads();
		for (atom2=0;atom2<nAtoms;atom2++) {
			dist2 =0.0;
			for (k=0;k<nDim;k++) {
				r[k] = xyz_shared[atom1*nDim+k] - xyz_shared[atom2*nDim+k];
				dist2 += r[k]*r[k];
			}
			pairDist[atom1*nAtoms+atom2] = dist2;
		}

	}
}

/* C wrappers for kernels */

extern "C" void pair_dist_cuda(float *xyz_d, float *pairDist_d, int nAtoms) {
	int blockSize;      // The launch configurator returned block size 
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    	int gridSize;       // The actual grid size needed, based on input size 
	unsigned int *blockID_d;
	unsigned int *blockID_h;
	int i;

	blockID_h = (unsigned int *)malloc( nAtoms*sizeof(unsigned int));
	cudaMalloc((void **) &blockID_d, nAtoms*sizeof(unsigned int));

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pair_dist_kernel, 0, nAtoms); 

    	// Round up according to array size 
    	gridSize = (nAtoms + blockSize - 1) / blockSize; 
	gridSize = 1;
	blockSize = nAtoms;
	printf("gridSize= %d blockSize= %d\n",gridSize,blockSize);
	// run nonbond cuda kernel
	pair_dist_kernel<<<gridSize, blockSize, nAtoms*nDim*sizeof(float)>>>(xyz_d, pairDist_d, nAtoms, blockID_d);
	// pass device variable, blockID_d, to host variable blockID_h
	cudaMemcpy(blockID_h, blockID_d, nAtoms*sizeof(unsigned int), cudaMemcpyDeviceToHost);	
//	for (i=0;i<nAtoms;i++) {
//		printf("%d\n", blockID_h[i]);
//	}

}

