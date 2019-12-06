#include "radonusfft.cuh"
#include "kernels_radonusfft.cuh"
#include <stdio.h>

radonusfft::radonusfft(size_t n_, size_t ntheta_, size_t nz_, float center_)
{
	n = n_;
	ntheta = ntheta_;
	nz = nz_;
	center = center_;
	float eps = 1e-3; // accuracy of USFFT
	mu = -log(eps) / (2 * n * n);
	m = ceil(2 * n * 1 / PI * sqrt(-mu * log(eps) + (mu * n) * (mu * n) / 4)); // interpolation radius according to accuracy
	cudaMalloc((void **)&f, n * n * nz * sizeof(float2));
	cudaMalloc((void **)&g, n * ntheta * nz * sizeof(float2));
	cudaMalloc((void **)&fde, (2 * n + 2 * m) * (2 * n + 2 * m) * nz * sizeof(float2));
	cudaMalloc((void **)&x, n * ntheta * sizeof(float));
	cudaMalloc((void **)&y, n * ntheta * sizeof(float));
	cudaMalloc((void **)&theta, ntheta * sizeof(float));

	int ffts[2];
	int idist;
	int odist;
	int inembed[2];
	int onembed[2];
	//fft 2d
	ffts[0] = 2 * n;
	ffts[1] = 2 * n;
	idist = (2 * n + 2 * m) * (2 * n + 2 * m);
	odist = (2 * n + 2 * m) * (2 * n + 2 * m);
	inembed[0] = 2 * n + 2 * m;
	inembed[1] = 2 * n + 2 * m;
	onembed[0] = 2 * n + 2 * m;
	onembed[1] = 2 * n + 2 * m;
	cufftPlanMany(&plan2d, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, nz);

	//fft 1d
	ffts[0] = n;
	idist = n;
	odist = n;
	inembed[0] = n;
	onembed[0] = n;
	cufftPlanMany(&plan1d, 1, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, ntheta * nz);
	
  	cudaMalloc((void **)&shiftfwd, n * sizeof(float2));
  	cudaMalloc((void **)&shiftadj, n * sizeof(float2));
  	// compute shifts with respect to the rotation center
  	takeshift <<<ceil(n / 1024.0), 1024>>> (shiftfwd, -(center - n / 2.0), n);
  	takeshift <<<ceil(n / 1024.0), 1024>>> (shiftadj, (center - n / 2.0), n);

	BS2d = dim3(32, 32);
	BS3d = dim3(32, 32, 1);

	GS2d0 = dim3(ceil(n / (float)BS2d.x), ceil(ntheta / (float)BS2d.y));
	GS3d0 = dim3(ceil(n / (float)BS3d.x), ceil(n / (float)BS3d.y), ceil(nz / (float)BS3d.z));
	GS3d1 = dim3(ceil(2 * n / (float)BS3d.x), ceil(2 * n / (float)BS3d.y), ceil(nz / (float)BS3d.z));
	GS3d2 = dim3(ceil((2 * n + 2 * m) / (float)BS3d.x), ceil((2 * n + 2 * m) / (float)BS3d.y), ceil(nz / (float)BS3d.z));
	GS3d3 = dim3(ceil(n / (float)BS3d.x), ceil(ntheta / (float)BS3d.y), ceil(nz / (float)BS3d.z));	
}

radonusfft::~radonusfft()
{
	cudaFree(f);
	cudaFree(g);
	cudaFree(fde);
	cudaFree(x);
	cudaFree(y);
	cudaFree(shiftfwd);
	cudaFree(shiftadj);
	cufftDestroy(plan2d);
	cufftDestroy(plan1d);
}

void radonusfft::fwdR(float2 *g_, float2 *f_, float *theta_, cudaStream_t s)
{	
	//NOTE: SIZE(g) = [nz,ntheta,n]
	cudaMemcpyAsync(f, f_, n * n * nz * sizeof(float2), cudaMemcpyDefault, s);
	cudaMemcpyAsync(theta, theta_, ntheta * sizeof(float), cudaMemcpyDefault, s);
	cudaMemsetAsync(fde, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * nz * sizeof(float2), s);

	takexy<<<GS2d0, BS2d, 0, s>>>(x, y, theta, n, ntheta);
	divphi<<<GS3d0, BS3d, 0, s>>>(fde, f, mu, m, n, nz, TOMO_FWD);
	
	fftshift<<<GS3d2, BS3d, 0, s>>>(fde, 2 * n + 2 * m,2 * n + 2 * m, nz, 1);
	cufftSetStream(plan2d, s);
	cufftExecC2C(plan2d, (cufftComplex *)&fde[m + m * (2 * n + 2 * m)], (cufftComplex *)&fde[m + m * (2 * n + 2 * m)], CUFFT_FORWARD);
	fftshift<<<GS3d2, BS3d, 0, s>>>(fde, 2 * n + 2 * m, 2 * n + 2 * m, nz, 1);
	
	wrap<<<GS3d2, BS3d, 0, s>>>(fde, n, nz, m, TOMO_FWD);
	gather<<<GS3d3, BS3d, 0, s>>>(g, fde, x, y, m, mu, n, ntheta, nz, TOMO_FWD);	
	
	// shift with respect to given center
  	shift <<<GS3d3, BS3d, 0, s>>> (g, shiftfwd, n, ntheta, nz);	  
	fftshift<<<GS3d3, BS3d, 0, s>>>(g, n, ntheta, nz, 0);
	cufftSetStream(plan1d, s);
	cufftExecC2C(plan1d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE);
	fftshift<<<GS3d3, BS3d, 0, s>>>(g, n, ntheta, nz, 0);
	
	mulr<<<GS3d3, BS3d, 0, s>>>(g, 1.0f / sqrt(n * ntheta), n, ntheta, nz);
	cudaMemcpyAsync(g_, g, n * ntheta * nz * sizeof(float2), cudaMemcpyDefault, s);
}

void radonusfft::adjR(float2 *f_, float2 *g_, float *theta_, bool filter, cudaStream_t s)
{
	//NOTE: SIZE(g) = [nz,ntheta,n]
	cudaMemcpyAsync(g, g_, n * ntheta * nz * sizeof(float2), cudaMemcpyDefault, s);
	cudaMemcpyAsync(theta, theta_, ntheta * sizeof(float), cudaMemcpyDefault, s);
	cudaMemsetAsync(fde, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * nz * sizeof(float2), s);

	takexy<<<GS2d0, BS2d, 0, s>>>(x, y, theta, n, ntheta);

	fftshift<<<GS3d3, BS3d, 0, s>>>(g, n, ntheta, nz, 0 );
	cufftSetStream(plan1d, s);
	cufftExecC2C(plan1d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
	fftshift<<<GS3d3, BS3d, 0, s>>>(g, n, ntheta, nz, 0);
	
	// shift with respect to given center	
	shift <<<GS3d3, BS3d, 0 , s>>> (g, shiftadj, n, ntheta, nz);
	gather<<<GS3d3, BS3d, 0, s>>>(g, fde, x, y, m, mu, n, ntheta, nz, TOMO_ADJ);	
	wrap<<<GS3d2, BS3d, 0, s>>>(fde, n, nz, m, TOMO_ADJ);
	
	fftshift<<<GS3d2, BS3d, 0, s>>>(fde, 2 * n + 2 * m, 2 * n + 2 * m, nz, 1);
	
	cufftSetStream(plan2d, s);
	cufftExecC2C(plan2d, (cufftComplex *)&fde[m + m * (2 * n + 2 * m)], (cufftComplex *)&fde[m + m * (2 * n + 2 * m)], CUFFT_INVERSE);
	fftshift<<<GS3d2, BS3d, 0, s>>>(fde, 2 * n + 2 * m, 2 * n + 2 * m, nz, 1);
	divphi<<<GS3d0, BS3d, 0, s>>>(fde, f, mu, m, n, nz, TOMO_ADJ);
	
	mulr<<<GS3d0, BS3d, 0, s>>>(f, 1.0f / sqrt(n * ntheta), n, n, nz);

	cudaMemcpyAsync(f_, f, n * n * nz * sizeof(float2), cudaMemcpyDefault, s);
}
