
#include "rectv.cuh"

__global__ void prox1(float *h1, float *g, float sigma, int N, int Ntheta, int Nz)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= N || ty >= Ntheta || tz >= Nz)
		return;

	int id0 = tx + ty * N + tz * N * Ntheta;
	h1[id0] = (h1[id0] - sigma * g[id0]) / (1 + sigma);
}

__global__ void prox2(float4 *h2, float lambda, int N, int M, int Nz)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z * blockDim.z + threadIdx.z;
	int tt = tz % M;
	tz = tz / M;
	if (tx >= N || ty >= N || tt >= M || tz >= Nz)
		return;

	int id0 = tx + ty * N + tt * N * N + tz * N * N * M;
	float no = max(1.0f, 1.0f / lambda * sqrtf(h2[id0].x * h2[id0].x + h2[id0].y * h2[id0].y + h2[id0].z * h2[id0].z + h2[id0].w * h2[id0].w));
	h2[id0].x /= no;
	h2[id0].y /= no;
	h2[id0].z /= no;
	h2[id0].w /= no;
}

void __global__ updateft_ker(float *ftn, float *fn, float *f, int N, int M, int Nz)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z * blockDim.z + threadIdx.z;
	int tt = tz % M;
	tz = tz / M;
	if (tx >= N || ty >= N || tt >= M || tz >= Nz)
		return;

	int id = tx + ty * N + tt * N * N + tz * N * N * M;
	ftn[id] = 2 * fn[id] - f[id];
}

void rectv::prox(float *h1, float4 *h2, float *g, float tau, int igpu, cudaStream_t s)
{
	
	prox1<<<GS3d2, BS3d, 0, s>>>(h1, g, tau, N, Ntheta, Nzp);
	prox2<<<GS3d4, BS3d, 0, s>>>(h2, lambda0, N + 1, M + 1, Nzp + 1);
}

void rectv::updateft(float *ftn, float *fn, float *f, float tau, int igpu, cudaStream_t s)
{
		updateft_ker<<<GS3d0, BS3d, 0, s>>>(ftn, fn, f, N, M, Nzp);
}

void rectv::solver_chambolle(float *f0, float *fn0, float *ft0, float *ftn0, float *h10, float4 *h20, float *g0,  int iz, int igpu, cudaStream_t s)
{
	float tau = 1 / sqrt(1 + 1 + lambda1 * lambda1 + (Nz != 0)); //sqrt norm of K1^*K1+K2^*K2	
	//forward step
	gradient(h20, ft0, tau, iz, igpu, s); //iz for border control
	radonapr(h10, ft0, tau, igpu, s);
	//proximal
	prox(h10, h20, g0, tau, igpu, s);
	//backward step
	divergent(fn0, f0, h20, tau, igpu, s);
	radonapradj(fn0, h10, tau, igpu, s);
	//update ft
	updateft(ftn0, fn0, f0, tau, igpu, s);	
}
