/*interface*/
%module rectv

%{
#define SWIG_FILE_WITH_INIT
#include "rectv.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}

class rectv
{
	//parameters
	size_t N;
	size_t Ntheta;
	size_t M;
	size_t Nz;
	size_t Nzp;
	float tau;
	float lambda0;
	float lambda1;

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
	float4 *psi;
	float4 *mu;
	float **theta;
	float2 **phi;

	//temporary arrays on gpus
	float **ftmp;
	float **gtmp;
	float **ftmps;
	float **gtmps;

	void radonapr(float *g, float *f, float tau,int igpu, cudaStream_t s);
	void radonapradj(float *f, float *g, float tau,int igpu, cudaStream_t s);
	void gradient(float4 *g, float *f, float tau,int iz, int igpu, cudaStream_t s);
	void divergent(float *fn, float *f, float4 *g,float tau, int igpu, cudaStream_t s);
	void prox(float *h1, float4 *h2, float *g,float tau, int igpu, cudaStream_t s);
	void updateft(float *ftn, float *fn, float *f, float tau,  int igpu, cudaStream_t s);	
	void updatemu(float4* mu, float4* h2, float4* psi, float rho, cudaStream_t s);
	void solver_chambolle(float *f0, float *fn0, float *ft0, float *ftn0, float *h10, float4 *h20, float *g0,  int iz, int igpu, cudaStream_t s);	
	void cg(float *f0, float *fn0, float *h10, float4 *h20, float *g0, float4* psi0, float4* mu0, float rho,  int iz, int niter, int igpu, cudaStream_t s);	
	void solver_admm(float *ft0, float *ftn0, float *h10, float4 *h20, float *g0, float4* psi0, float4* mu0, int iz, int igpu, cudaStream_t s);	
	void solve_reg(float4* psi, float4* h2, float4* mu, float lambda, float rho, cudaStream_t s);
public:
	rectv(size_t N, size_t Ntheta, size_t M, size_t Nz, size_t Nzp,
		  size_t ngpus, float center, float lambda0, float lambda1);
	~rectv();
	void run(float *fres, float *g, float* theta, float* phi, size_t niter);
	void adjoint_tests(float *g, float* theta, float* phi);

// wrappers for python interface
%apply(float *INPLACE_ARRAY1, int DIM1){(float *fres, int N0)};
%apply(float *IN_ARRAY1, int DIM1){(float *g, int N1)};
%apply(float *IN_ARRAY1, int DIM1){(float *theta, int N2)};
%apply(float *IN_ARRAY1, int DIM1){(float *phi, int N3)};
	void run_wrap(float *fres, int N0, float *g, int N1, float *theta, int N2, float *phi, int N3, size_t niter);
	void adjoint_tests_wrap(float *g, int N1, float *theta, int N2, float *phi, int N3);
};
