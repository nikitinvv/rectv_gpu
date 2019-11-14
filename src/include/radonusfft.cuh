#pragma once
#include <cufft.h>

class radonusfft
{
	size_t N;
	size_t Ntheta;
	size_t Nz;
	size_t M;
	float mu;

	float2 *f;
	float2 *g;
	float *theta;

	float *x;
	float *y;

	float2 *fde;

	cufftHandle plan2d;
	cufftHandle plan1d;

public:
	radonusfft(size_t N, size_t Ntheta, size_t Nz);
	~radonusfft();
	void fwdR(float2 *g, float2 *f, float *theta, cudaStream_t s);
	void adjR(float2 *f, float2 *g, float *theta, bool filter, cudaStream_t s);
};
