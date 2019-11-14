#include "radonusfft.cuh"
#include "kernels_radonusfft.cuh"
#include <stdio.h>

radonusfft::radonusfft(size_t N_, size_t Ntheta_, size_t Nz_)
{
	N = N_;
	Ntheta = Ntheta_;
	Nz = Nz_;
	float eps = 1e-3;
	mu = -log(eps)/(2*N*N);
	M = ceil(2*N*1/PI*sqrt(-mu*log(eps)+(mu*N)*(mu*N)/4));
	cudaMalloc((void**)&f,N*N*Nz*sizeof(float2));
	cudaMalloc((void**)&g,N*Ntheta*Nz*sizeof(float2));
	cudaMalloc((void**)&fde,(2*N+2*M)*(2*N+2*M)*Nz*sizeof(float2));
	cudaMalloc((void**)&x,N*Ntheta*sizeof(float));
	cudaMalloc((void**)&y,N*Ntheta*sizeof(float));
	cudaMalloc((void**)&theta,Ntheta*sizeof(float));

	int ffts[2];
	int idist;int odist;
	int inembed[2];int onembed[2];
	//fft 2d 
	ffts[0] = 2*N; ffts[1] = 2*N;
	idist = (2*N+2*M)*(2*N+2*M);odist = (2*N+2*M)*(2*N+2*M);
	inembed[0] = 2*N+2*M; inembed[1] = 2*N+2*M;
	onembed[0] = 2*N+2*M; onembed[1] = 2*N+2*M;
	cufftPlanMany(&plan2d, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, Nz); 

	//fft 1d	
	ffts[0] = N;
	idist = N;odist = N;
	inembed[0] = N;onembed[0] = N;
	cufftPlanMany(&plan1d, 1, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, Ntheta*Nz);
}

radonusfft::~radonusfft()
{	
	cudaFree(f);
	cudaFree(g);
	cudaFree(fde);
	cudaFree(x);
	cudaFree(y);
	cudaFree(theta);
	cufftDestroy(plan2d);
	cufftDestroy(plan1d);
}

void radonusfft::fwdR(float2* g_, float2* f_, float* theta_, cudaStream_t s)
{	
	dim3 BS2d(32,32);
	dim3 BS3d(32,32,1);

	dim3 GS2d0(ceil(N/(float)BS2d.x),ceil(Ntheta/(float)BS2d.y));
	dim3 GS3d0(ceil(N/(float)BS3d.x),ceil(N/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d1(ceil(2*N/(float)BS3d.x),ceil(2*N/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d2(ceil((2*N+2*M)/(float)BS3d.x),ceil((2*N+2*M)/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d3(ceil(N/(float)BS3d.x),ceil(Ntheta/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	cudaMemcpyAsync(f,f_,N*N*Nz*sizeof(float2),cudaMemcpyDefault,s);
	cudaMemcpyAsync(theta,theta_,Ntheta*sizeof(float),cudaMemcpyDefault,s);  	
	cudaMemsetAsync(fde,0,(2*N+2*M)*(2*N+2*M)*Nz*sizeof(float2),s);
	takexy<<<GS2d0, BS2d,0,s>>>(x,y,theta,N,Ntheta);
	divphi<<<GS3d0, BS3d,0,s>>>(fde,f,mu,M,N,Nz);

	fftshiftc<<<GS3d2, BS3d,0,s>>>(fde,2*N+2*M,Nz);
	cufftSetStream(plan2d,s);
	cufftExecC2C(plan2d, (cufftComplex*)&fde[M+M*(2*N+2*M)],(cufftComplex*)&fde[M+M*(2*N+2*M)],CUFFT_FORWARD);
	fftshiftc<<<GS3d2, BS3d,0,s>>>(fde,2*N+2*M,Nz);

	wrap<<<GS3d2, BS3d,0,s>>>(fde,N,Nz,M);
	gather<<<GS3d3, BS3d,0,s>>>(g,fde,x,y,M,mu,N,Ntheta,Nz);

	fftshift1c<<<GS3d3, BS3d,0,s>>>(g,N,Ntheta,Nz);
	cufftSetStream(plan1d,s);
	cufftExecC2C(plan1d, (cufftComplex*)g,(cufftComplex*)g,CUFFT_INVERSE);
	fftshift1c<<<GS3d3, BS3d,0,s>>>(g,N,Ntheta,Nz);

	mulr<<<GS3d3,BS3d,0,s>>>(g,1.0f/(4*N*N*N*sqrt(N*Ntheta)),N,Ntheta,Nz);
	cudaMemcpyAsync(g_,g,N*Ntheta*Nz*sizeof(float2),cudaMemcpyDefault,s);  	
}

void radonusfft::adjR(float2* f_, float2* g_, float* theta_, bool filter, cudaStream_t s)
{
	dim3 BS2d(32,32);
	dim3 BS3d(32,32,1);

	dim3 GS2d0(ceil(N/(float)BS2d.x),ceil(Ntheta/(float)BS2d.y));
	dim3 GS3d0(ceil(N/(float)BS3d.x),ceil(N/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d1(ceil(2*N/(float)BS3d.x),ceil(2*N/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d2(ceil((2*N+2*M)/(float)BS3d.x),ceil((2*N+2*M)/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d3(ceil(N/(float)BS3d.x),ceil(Ntheta/(float)BS3d.y),ceil(Nz/(float)BS3d.z));

	cudaMemcpyAsync(g,g_,N*Ntheta*Nz*sizeof(float2),cudaMemcpyDefault,s);
	cudaMemcpyAsync(theta,theta_,Ntheta*sizeof(float),cudaMemcpyDefault,s);  	

	cudaMemsetAsync(fde,0,(2*N+2*M)*(2*N+2*M)*Nz*sizeof(float2),s);

	takexy<<<GS2d0, BS2d,0,s>>>(x,y,theta,N,Ntheta);

	fftshift1c<<<GS3d3, BS3d,0,s>>>(g,N,Ntheta,Nz);
	cufftSetStream(plan1d,s);
	cufftExecC2C(plan1d, (cufftComplex*)g,(cufftComplex*)g,CUFFT_FORWARD);
	fftshift1c<<<GS3d3, BS3d,0,s>>>(g,N,Ntheta,Nz);

	if(filter) applyfilter<<<GS3d3, BS3d,0,s>>>(g,N,Ntheta,Nz);

	scatter<<<GS3d3, BS3d,0,s>>>(fde,g,x,y,M,mu,N,Ntheta,Nz);
	wrapadj<<<GS3d2, BS3d,0,s>>>(fde,N,Nz,M);

	fftshiftc<<<GS3d2, BS3d,0,s>>>(fde,2*N+2*M,Nz);
	cufftSetStream(plan2d,s);
	cufftExecC2C(plan2d, (cufftComplex*)&fde[M+M*(2*N+2*M)],(cufftComplex*)&fde[M+M*(2*N+2*M)],CUFFT_INVERSE);
	fftshiftc<<<GS3d2, BS3d,0,s>>>(fde,2*N+2*M,Nz);

	unpaddivphi<<<GS3d0, BS3d,0,s>>>(f,fde,mu,M,N,Nz);
	mulr<<<GS3d0,BS3d,0,s>>>(f,1.0f/(4*N*N*N*sqrt(N*Ntheta)),N,N,Nz);

	cudaMemcpyAsync(f_,f,N*N*Nz*sizeof(float2),cudaMemcpyDefault,s);
}

