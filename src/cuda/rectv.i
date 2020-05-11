/*interface*/
%module rectv

%{
#define SWIG_FILE_WITH_INIT
#include "rectv.cuh"
%}

class rectv
{

	
public:
	%immutable;
	int n;
	int ntheta;
	int m;
	int nz;
	
	%mutable;
	rectv(int n, int ntheta, int m, int nz, int nzp, int ngpus);
	~rectv();
	void run(size_t fres, size_t g, size_t theta, size_t phi, 
		float center, float lambda0, float lambda1, float step, 
		int niter, int titer, bool dbg);
	void check_adjoints(size_t res, size_t g, size_t theta, size_t phi, float lambda1,float center);
	void free();
};
