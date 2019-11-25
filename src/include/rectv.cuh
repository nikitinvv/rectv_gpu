#include <cuda_runtime.h>
#include "radonusfft.cuh"

class rectv
{
	//parameters
	int n;
	int ntheta;
	int m;
	int nz;
	int nzp;
	float lambda0;
	float lambda1;

	dim3 BS2d, BS3d, GS2d0, GS3d0, GS3d1, GS3d2, GS3d3, GS3d4;

	//number of gpus
	int ngpus;

	//class for applying Radon transform
	radonusfft **rad;

	//vars
	float *f;
	float *fn;
	float *g;
	float4 *psi;
	float4 *mu;
	float **theta;
	float2 **phi;

	//temporary arrays on gpus
	float **ftmp;
	float **gtmp;
	float **ftmps;
	float **gtmps;
	float **fm;
	float **h1;
	float4 **h2;
	
	void radonapr(float *g, float *f, float tau, int igpu, cudaStream_t s);
	void radonapradj(float *f, float *g, float tau, int igpu, cudaStream_t s);
	void gradient(float4 *g, float *f, int iz, int igpu, cudaStream_t s);
	void divergent(float *fn, float4 *g, float tau, int igpu, cudaStream_t s);
	void solver_admm(float *f, float *fn, float *h1, float4 *h2, float* fm, float *g, float4 *psi, float4 *mu, int iz, int titer, int igpu, cudaStream_t s);
	
public:
	rectv(int n, int ntheta, int m, int nz, int nzp,
		  int ngpus, float center, float lambda0, float lambda1);
	~rectv();
	// Reconstruction by the Chambolle-Pock algorithm with proximal operators
	void run(size_t fres, size_t g, size_t theta, size_t phi, int niter, int titer);
	void adjoint_tests(size_t g, size_t theta, size_t phi);
};
