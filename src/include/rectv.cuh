#include <cuda_runtime.h>
#include "radonusfft.cuh"

class rectv
{
	//parameters
	size_t n;
	size_t ntheta;
	size_t m;
	size_t nz;
	size_t nzp;
	float lambda0;
	float lambda1;

	dim3 BS2d, BS3d, GS2d0, GS3d0, GS3d1, GS3d2, GS3d3, GS3d4;

	//number of gpus
	size_t ngpus;

	//class for applying Radon transform
	radonusfft **rad;

	//vars
	float *ft;
	float *ftn;
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
	float **fe;
	float **h1;
	float4 **h2;
	
	void radonapr(float *g, float *f, float tau, int igpu, cudaStream_t s);
	void radonapradj(float *f, float *g, float tau, int igpu, cudaStream_t s);
	void gradient(float4 *g, float *f, float tau, int iz, int igpu, cudaStream_t s);
	void divergent(float *fn, float *f, float4 *g, float tau, int igpu, cudaStream_t s);
	void updatemu(float4* mu, float4* h2, float4* psi, float rho, cudaStream_t s);
	void cg(float *f0, float *fn0, float *h10, float4 *h20, float *g0, float4 *psi0, float4 *mu0, float rho, int iz, int titer, int igpu, cudaStream_t s);
	void solver_admm(float *ft0, float *ftn0, float *h10, float4 *h20, float *g0, float4 *psi0, float4 *mu0, int iz, int titer, int igpu, cudaStream_t s);
	void solve_reg(float4* psi, float4* h2, float4* mu, float lambda, float rho, cudaStream_t s);

public:
	rectv(size_t n, size_t ntheta, size_t m, size_t nz, size_t nzp,
		  size_t ngpus, float center, float lambda0, float lambda1);
	~rectv();
	// Reconstruction by the Chambolle-Pock algorithm with proximal operators
	void run(float *fres, float *g, float *theta, float *phi, size_t niter, size_t titer);
	void adjoint_tests(float *g, float *theta, float *phi);
	// wrappers for python interface
	void run_wrap(float *fres, int n0, float *g, int n1, float *theta, int n2, float *phi, int n3, size_t niter, size_t titer);
	void adjoint_tests_wrap(float *g, int n1, float *theta, int n2, float *phi, int n3);
};
