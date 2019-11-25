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
	int n;
	int ntheta;
	int m;
	int nz;
	int nzp;
	float tau;
	float lambda0;
	float lambda1;

	//number of gpus
	int ngpus;

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

	

	void radonapr(float *g, float *f, float tau,int igpu, cudaStream_t s);
	void radonapradj(float *f, float *g, float tau,int igpu, cudaStream_t s);
	void gradient(float4 *g, float *f, int iz, int igpu, cudaStream_t s);
	void divergent(float *fn, float4 *g, float tau, int igpu, cudaStream_t s);
	void solver_admm(float *f, float *fn, float *h1, float4 *h2, float *g, float4* psi, float4* mu, int iz, int titer, int igpu, cudaStream_t s);	
public:
	rectv(int n, int ntheta, int m, int nz, int nzp,
		  int ngpus, float center, float lambda0, float lambda1);
	~rectv();
	
	void run(size_t fres, size_t g_, size_t theta_, size_t phi_, int niter, int titer);
	void adjoint_tests(size_t g, size_t theta, size_t phi);
};
