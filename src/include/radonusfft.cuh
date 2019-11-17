#pragma once
#include <cufft.h>

enum dir {
  TOMO_FWD,
  TOMO_ADJ
};

class radonusfft
{
	size_t N;
	size_t Ntheta;
	size_t Nz;
	float center;
	size_t M;
	float mu;

	float2 *f;
	float2 *g;
	float *theta;
	
	float *x;
	float *y;
	float2 *fde;

	float2* shiftfwd;
	float2* shiftadj;
	cufftHandle plan2d;
	cufftHandle plan1d;
	
	dim3 BS2d, BS3d, GS2d0, GS3d0, GS3d1, GS3d2, GS3d3;

public:
	radonusfft(size_t N, size_t Ntheta, size_t Nz, float center);
	~radonusfft();
	void fwdR(float2 *g, float2 *f, float *theta, cudaStream_t s);
	void adjR(float2 *f, float2 *g, float *theta, bool filter, cudaStream_t s);
};
