#include <cuda_runtime.h>
#include "radonusfft.cuh"

class rectv
{
	//parameters
	size_t N;
	size_t Ntheta;
	size_t M;
	size_t Nz;
	size_t Nzp;
	float lambda0;
	float lambda1;

	dim3 BS2d, BS3d, GS2d0, GS3d0, GS3d1, GS3d2, GS3d3, GS3d4;

	//number of gpus
	size_t ngpus;

	//class for applying Radon transform
	radonusfft **rad;

	//vars
	float *f;
	float *fn;
	float *ft;
	float *ftn;
	float *g;
	float *h1;
	float4 *h2;
	float **theta;
	float2 **phi;

	//temporary arrays on gpus
	float **ftmp;
	float **gtmp;
	float **ftmps;
	float **gtmps;

	void radonapr(float *g, float *f, float tau, int igpu, cudaStream_t s);
	void radonapradj(float *f, float *g, float tau, int igpu, cudaStream_t s);
	void gradient(float4 *g, float *f, float tau, int iz, int igpu, cudaStream_t s);
	void divergent(float *fn, float *f, float4 *g, float tau, int igpu, cudaStream_t s);
	void prox(float *h1, float4 *h2, float *g, float tau, int igpu, cudaStream_t s);
	void updateft(float *ftn, float *fn, float *f, int igpu, cudaStream_t s);
	void solver_chambolle(float *f0, float *fn0, float *ft0, float *ftn0, float *h10, float4 *h20, float *g0, int iz, int igpu, cudaStream_t s);
	
	
public:
	rectv(size_t N, size_t Ntheta, size_t M, size_t Nz, size_t Nzp,
		  size_t ngpus, float center, float lambda0, float lambda1);
	~rectv();
	// Reconstruction by the Chambolle-Pock algorithm with proximal operators
	void run(float *fres, float *g, float *theta, float *phi, size_t niter);
	void adjoint_tests(float *g, float* theta, float* phi);
	// wrappers for python interface
	void run_wrap(float *fres, int N0, float *g, int N1, float *theta, int N2, float *phi, int N3, size_t niter);
	void adjoint_tests_wrap(float *g, int N1, float *theta, int N2, float *phi, int N3);

};
