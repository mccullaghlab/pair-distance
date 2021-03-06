
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
#include "atom_class.h"
#define nDim 3

void atom::initialize(float T, float lbox, int nMC)
{
	float dist2, temp;
	int igo;
	float sigma2;
	// atoms and types
	nAtoms = 100;
	nAtomTypes = 1;
	numNNmax = 200;
	// size of xyz arrays
	nAtomBytes = nAtoms*sizeof(float);
	nTypeBytes = nAtomTypes*sizeof(float);
	// allocate atom coordinate arrays
	xyz_h = (float *)malloc(nAtomBytes*nDim);
	// allocate atom velocity arrays
	v_h = (float *)malloc(nAtomBytes*nDim);
	// allocate atom force arrays
	f_h = (float *)malloc(nAtomBytes*nDim);
	// alocate mass array
	mass_h = (float *)malloc(nAtoms*sizeof(float));
	// alocate charge array
	charges_h = (float *)malloc(nAtoms*sizeof(float));
	// allocate key array - atom number
	key = (int *)malloc(nAtoms*sizeof(int));
	// allocate atom type arrays
	ityp_h = (int *)malloc(nAtoms*sizeof(int));
	// allocate atom based parameter arrays
	x0_h = (float *)malloc(nTypeBytes);
	g0_h = (float *)malloc(nTypeBytes);
	gr2_h = (float *)malloc(nTypeBytes*2);
	w_h = (float *)malloc(nTypeBytes);
	alpha_h = (float *)malloc(nTypeBytes);
	vtot_h = (float *)malloc(nTypeBytes);
	lj_A_h = (float *)malloc(nTypeBytes);
	lj_B_h = (float *)malloc(nTypeBytes);
	
	// populate host arrays
	gr2_h[0] = 11.002;
	gr2_h[1] = 21.478;
	w_h[0] = 0.801;
	g0_h[0] = 1.714; // height of parabola
	x0_h[0] = 4.118;
	alpha_h[0] = 2.674; 
	lj_A_h[0] = 6.669e7;
	lj_B_h[0] = 1.103e4;
	vtot_h[0] = 16.0/3.0*3.1415926535*w_h[0]*g0_h[0]/((float) nMC)*0.0334*1E-2;
	sigma = pow(lj_A_h[0]/lj_B_h[0],(1.0/6.0));
	sigma2 = sigma*sigma;

	for (i=0;i<nAtoms;i++) {
//		xyz_h[i*nDim] = (float) i*7.0;
//		xyz_h[i*nDim+1] = xyz_h[i*nDim+2] = 0.0f;
		key[i] = i;
		f_h[i*nDim] = f_h[i*nDim+1] = f_h[i*nDim+2] = 0.0f;
		ityp_h[i] = 0;
		charges_h[i] = 0.0;
		mass_h[i] = 12.0;
		for (k=0;k<nDim;k++) {
			v_h[i*nDim+k] = rand_gauss()*sqrt(T/mass_h[i]);	
		}
		igo = 1;
		while (igo == 1) {
			igo = 0;
			for (k=0;k<nDim;k++) {
				xyz_h[i*nDim+k] = lbox*(float) rand() / (float) RAND_MAX;
			}
			for (j=0;j<i;j++) {
				dist2 = 0.0;
				for (k=0;k<nDim;k++) {
					temp = xyz_h[i*nDim+k] - xyz_h[j*nDim+k];
					if (temp > lbox/2.0) {
						temp -= lbox;
					} else if (temp < -lbox/2.0) {
						temp += lbox;
					}
					dist2 += temp*temp;
				}
				if (dist2 < sigma2) {
					igo = 1;
					break;
				}

			}
		}
	}

	// open files for printing later
	forceXyzFile = fopen("forces.xyz","w");
	xyzFile = fopen("positions.xyz","w");
	vFile = fopen("velocities.xyz","w");

}

float atom::rand_gauss() 
{
	
	float v1, v2, r2, fac;

	v1 = 1.0 - 2.0 * (float) rand() / (float) RAND_MAX;
	v2 = 1.0 - 2.0 * (float) rand() / (float) RAND_MAX;
	r2 = v1*v1 + v2*v2;
	while (r2 > 1.0) {
		v1 = 1.0 - 2.0 * (float) rand() / (float) RAND_MAX;
        	v2 = 1.0 - 2.0 * (float) rand() / (float) RAND_MAX;
		r2 = v1*v1 + v2*v2;
	}
	fac = sqrt(-2.0*log(r2)/r2);
	return v1*fac;
}

void atom::initialize_gpu()
{
	// allocate atom coordinate arrays
	cudaMalloc((void **) &xyz_d, nAtomBytes*nDim);
	cudaMalloc((void **) &pairDist_d, nAtomBytes*nAtomBytes);
	// allocate atom velocity arrays
	cudaMalloc((void **) &v_d, nAtomBytes*nDim);
	// allocate atom force arrays
	cudaMalloc((void **) &f_d, nAtomBytes*nDim);
	// allocate mass array
	cudaMalloc((void **) &mass_d, nAtomBytes);
	// allocate charges array
	cudaMalloc((void **) &charges_d, nAtomBytes);
	// allocate neighborlist stuff
	cudaMalloc((void **) &numNN_d, nAtoms*sizeof(int));
	cudaMalloc((void **) &NN_d, nAtoms*numNNmax*sizeof(int));
	// allocate atom type arrays
	cudaMalloc((void **) &ityp_d, nAtoms*sizeof(int));
	// allocate atom based parameter arrays
	cudaMalloc((void **) &x0_d, nTypeBytes);
	cudaMalloc((void **) &g0_d, nTypeBytes);
	cudaMalloc((void **) &gr2_d, nTypeBytes*2);
	cudaMalloc((void **) &w_d, nTypeBytes);
	cudaMalloc((void **) &alpha_d, nTypeBytes);
	cudaMalloc((void **) &vtot_d, nTypeBytes);
	cudaMalloc((void **) &lj_A_d, nTypeBytes);
	cudaMalloc((void **) &lj_B_d, nTypeBytes);

	cudaMalloc((void **) &numNN_d, nAtoms*sizeof(int));
	// the following will change depending on max density
	cudaMalloc((void **) &NN_d, nAtoms*nAtoms*sizeof(int));
}	


// copy parameter arrays to GPU
void atom::copy_params_to_gpu() {

	// copy data to device
	cudaMemcpy(ityp_d, ityp_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(w_d, w_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(x0_d, x0_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(g0_d, g0_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(gr2_d, gr2_h, 2*nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(alpha_d, alpha_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(vtot_d, vtot_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(lj_A_d, lj_A_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(lj_B_d, lj_B_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(mass_d, mass_h, nAtoms*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(charges_d, charges_h, nAtoms*sizeof(float), cudaMemcpyHostToDevice);	
}
// copy position, force and velocity arrays to GPU
void atom::copy_pos_v_to_gpu() {
	cudaMemcpy(v_d, v_h, nAtomBytes*nDim, cudaMemcpyHostToDevice);	
	cudaMemcpy(xyz_d, xyz_h, nAtomBytes*nDim, cudaMemcpyHostToDevice);	
}
// copy position, force, and velocity arrays from GPU
void atom::get_pos_f_v_from_gpu() {
	// pass device variable, f_d, to host variable f_h
	cudaMemcpy(f_h, f_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
	// pass device variable, xyz_d, to host variable xyz_h
	cudaMemcpy(xyz_h, xyz_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
	// pass device variable, v_d, to host variable v_h
	cudaMemcpy(v_h, v_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
}
// copy position, and velocity arrays from GPU
void atom::get_pos_v_from_gpu() {
	// pass device variable, f_d, to host variable f_h
	cudaMemcpy(v_h, v_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
	// pass device variable, xyz_d, to host variable xyz_h
	cudaMemcpy(xyz_h, xyz_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
}

void atom::print_forces() {
	int ip;
	fprintf(forceXyzFile,"%d\n", nAtoms);
	fprintf(forceXyzFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
		ip = key[i];
		fprintf(forceXyzFile,"C %10.6f %10.6f %10.6f\n", f_h[ip*nDim],f_h[ip*nDim+1],f_h[ip*nDim+2]);
	}
	fflush(forceXyzFile);
}

void atom::print_xyz() {
	int ip;
	fprintf(xyzFile,"%d\n", nAtoms);
	fprintf(xyzFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
		ip = key[i];
		fprintf(xyzFile,"C %10.6f %10.6f %10.6f\n", xyz_h[ip*nDim], xyz_h[ip*nDim+1], xyz_h[ip*nDim+2]);
	}
	fflush(xyzFile);
}

void atom::print_v() {
	int ip;
	fprintf(vFile,"%d\n", nAtoms);
	fprintf(vFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
		ip = key[i];
		fprintf(vFile,"C %10.6f %10.6f %10.6f\n", v_h[ip*nDim], v_h[ip*nDim+1], v_h[ip*nDim+2]);
	}
	fflush(vFile);
}

	
void atom::free_arrays() {
	// free host variables
	free(key);
	free(xyz_h);
	free(f_h); 
	free(ityp_h); 
	free(w_h); 
	free(g0_h); 
	free(gr2_h); 
	free(x0_h); 
	free(alpha_h); 
	free(vtot_h); 
	free(lj_A_h); 
	free(lj_B_h); 
	free(charges_h); 
	fclose(forceXyzFile);
	fclose(xyzFile);
	fclose(vFile);
}

void atom::free_arrays_gpu() {
	// free device variables
	cudaFree(xyz_d); 
	cudaFree(pairDist_d); 
	cudaFree(f_d); 
	cudaFree(ityp_d); 
	cudaFree(w_d); 
	cudaFree(g0_d); 
	cudaFree(gr2_d); 
	cudaFree(x0_d); 
	cudaFree(alpha_d); 
	cudaFree(vtot_d); 
	cudaFree(lj_A_d); 
	cudaFree(lj_B_d); 
	cudaFree(charges_d); 
	cudaFree(numNN_d);
	cudaFree(NN_d);
}
