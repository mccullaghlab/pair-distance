
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "pair_dist_cuda.h"
#include "atom_class.h"
#include "config_class.h"

#define nDim 3
#define MC 10

using namespace std;

int main(void)  
{
	cudaEvent_t start, stop;
	float milliseconds;
	float day_per_millisecond;
	atom atoms;
	config configs;
	int i;
	int step;
	int *NN_h, *numNN_h;
	long long seed;
	seed = 0;
	
	NN_h = (int *)malloc(atoms.nAtoms*atoms.numNNmax*sizeof(int));
	numNN_h = (int *)malloc(atoms.nAtoms*sizeof(int));


	// initialize
	configs.initialize();
	atoms.initialize(configs.T, configs.lbox, configs.nMC);
	atoms.initialize_gpu();

	// start device timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	atoms.copy_params_to_gpu();
	atoms.copy_pos_v_to_gpu();

	// compute the neighborlist
	pair_dist_cuda(atoms.xyz_d, atoms.pairDist_d, atoms.nAtoms);



	// get GPU time
	cudaEventRecord(stop);
    	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed time = %20.10f ms\n", milliseconds);
	day_per_millisecond = 1e-3 /60.0/60.0/24.0;
//printf("Average ns/day = %20.10f\n", configs.nSteps*2e-6/(milliseconds*day_per_millisecond) );	
	
	// free up arrays
	atoms.free_arrays();
	atoms.free_arrays_gpu();

	return 0;

}


