#include <cuda_runtime.h>
#include "radonusfft.cuh"
#include "cublas_v2.h"
class rectv
{
	bool is_free;
	int nzp;	
	dim3 BS2d, BS3d, GS2d0, GS3d0, GS3d1, GS3d2, GS3d3, GS3d4;
	//number of gpus
	int ngpus;

	//class for applying Radon transform
	radonusfft **rad;
	cublasHandle_t* cublas_handles;
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
	float4 **h2stored;
	
	void radonapr(float *g, float *f, float tau, int igpu, cudaStream_t s);
	void radonapradj(float *f, float *g, float tau, int igpu, cudaStream_t s);
	void gradient(float4 *g, float *f, float lambda1, int iz, int igpu, cudaStream_t s);
	void divergent(float *fn, float4 *g, float lambda1, float tau, int igpu, cudaStream_t s);
	float2 solver_admm(float *f, float *fn, float *h1, float4 *h2, float4* h2stored, float* fm, float *g, float4 *psi, float4 *mu, 
		float lambda0, float lambda1, float rho, 
		int iz, int titer, int igpu, cudaStream_t s);
	void set_center(float center);
	
public:
	int n;
	int ntheta;
	int m;
	int nz;
	rectv(int n, int ntheta, int m, int nz, int nzp, int ngpus);
	~rectv();
	void run(size_t fres, size_t g, size_t theta, size_t phi, 
		float center, float lambda0, float lambda1, 
		int niter, int titer);
	void check_adjoints(size_t res,size_t g, size_t theta, size_t phi, float lambda1, float center);
	void free();
};
