
#include "rectv.cuh"
#include "stdio.h"

__global__ void diff(float *h1, float *g, int N, int Ntheta, int Nz)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= N || ty >= Ntheta || tz >= Nz)
        return;

    int id0 = tx + ty * N + tz * N * Ntheta;
    h1[id0] = (h1[id0] - g[id0]);
}

__global__ void updatemu_ker(float4 *mu, float4 *h2, float4 *psi, float rho, int N, int Nz)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= N || ty >= N || tz >= Nz)
        return;

    int id0 = tx + ty * N + tz * N * N;
    mu[id0].x += rho * (h2[id0].x - psi[id0].x);
    mu[id0].y += rho * (h2[id0].y - psi[id0].y);
    mu[id0].z += rho * (h2[id0].z - psi[id0].z);
    mu[id0].w += rho * (h2[id0].w - psi[id0].w);
}

__global__ void diffgrad(float4 *h2, float4 *psi, float4 *mu, float rho, int N, int Nz)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= N || ty >= N || tz >= Nz)
        return;

    int id0 = tx + ty * N + tz * N * N;
    h2[id0].x = rho * (h2[id0].x - psi[id0].x + mu[id0].x / rho);
    h2[id0].y = rho * (h2[id0].y - psi[id0].y + mu[id0].y / rho);
    h2[id0].z = rho * (h2[id0].z - psi[id0].z + mu[id0].z / rho);
    h2[id0].w = rho * (h2[id0].w - psi[id0].w + mu[id0].w / rho);
}

__global__ void solve_reg_ker(float4* psi, float4 *h2, float4* mu, float lambda, float rho, int N, int M, int Nz)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z * blockDim.z + threadIdx.z;
	int tt = tz % M;
	tz = tz / M;
	if (tx >= N || ty >= N || tt >= M || tz >= Nz)
		return;

	int id0 = tx + ty * N + tt * N * N + tz * N * N * M;
    psi[id0].x =(h2[id0].x+mu[id0].x/rho);
    psi[id0].y =(h2[id0].y+mu[id0].y/rho);
    psi[id0].z =(h2[id0].z+mu[id0].z/rho);
    psi[id0].w =(h2[id0].w+mu[id0].w/rho);    
	float za = sqrtf(psi[id0].x * psi[id0].x + psi[id0].y * psi[id0].y + psi[id0].z * psi[id0].z + psi[id0].w * psi[id0].w);
	if (za <= lambda / rho)
	{
		psi[id0].x = 0;
		psi[id0].y = 0;
		psi[id0].z = 0;
		psi[id0].w = 0;
	}
	else
	{
		psi[id0].x -= lambda / rho * psi[id0].x / za;
		psi[id0].y -= lambda / rho * psi[id0].y / za;
		psi[id0].z -= lambda / rho * psi[id0].z / za;
		psi[id0].w -= lambda / rho * psi[id0].w / za;
	}
}

void rectv::cg(float *ft0, float *ftn0, float *h10, float4 *h20, float *g0, float4 *psi0, float4 *mu0, float rho, int iz, int niter, int igpu, cudaStream_t s)       
{
    // cudaMemcpyAsync(ftn0, ft0, N * N * Nzp * M * sizeof(float), cudaMemcpyDefault, s);
    float* t1 = &ftn0[-(iz != 0) * N * N * M];
    float* t2 = &ft0[-(iz != 0) * N * N * M];
    cudaMemcpy(t1, t2,N*N*(Nzp + 2 - (iz == 0) - (iz == Nz / Nzp - 1))*M* sizeof(float), cudaMemcpyDefault); //mem+=N*N*M*(Nzp+2-(iz==0)-(iz==Nz/Nzp-1))*sizeof(float);
    
    
    // for( int i=0;i<N*N*(Nzp + 2 - (iz == 0) - (iz == Nz / Nzp - 1))*M;i++) norm+=ft0[-(iz != 0) * N * N * M+i];
    // printf("%f\n",norm);
    // // ft0 = ftn0;//do not change ft0
    for (int k=0;k<niter;k++)
    {
        cudaMemsetAsync(h10, 0, N * Ntheta * Nzp * sizeof(float), s);
        cudaMemsetAsync(h20, 0, (N + 1) * (N + 1) * (M + 1) * (Nzp + 1) * sizeof(float4), s);        
        //forward step
        // gradient(h20, ftn0, 1, iz, igpu, s); //iz for border control
        radonapr(h10, ftn0, 1, igpu, s);        
        cudaDeviceSynchronize();
        double norm=0;
        for( int i=0;i<N*Ntheta*Nzp;i++) norm+=h10[i];
        printf("%f\n",norm);
        //differences
        // diffgrad<<<GS3d4, BS3d, 0, s>>>(h20, psi0, mu0, rho, (N + 1), (M + 1) * (Nzp + 1));
        diff<<<GS3d2, BS3d, 0, s>>>(h10, g0, N, Ntheta, Nzp);
        //divergent(ftn0, ftn0, h20, 0.5, igpu, s);
        //backward step
        radonapradj(ftn0, h10, 0.5, igpu, s);         
    }    
    cudaMemsetAsync(h20, 0, (N + 1) * (N + 1) * (M + 1) * (Nzp + 1) * sizeof(float4), s);
}

void rectv::solve_reg(float4* psi, float4* h2, float4* mu, float lambda, float rho, cudaStream_t s)
{
    solve_reg_ker<<<GS3d4, BS3d, 0, s>>>(psi, h2, mu, lambda, rho, N, M, Nz);    
}

void rectv::updatemu(float4* mu, float4* h2, float4* psi, float rho, cudaStream_t s)
{
    updatemu_ker<<<GS3d4, BS3d, 0, s>>>(mu, psi, h2, rho, N, Nz);    
}

void rectv::solver_admm(float *ft0, float *ftn0, float *h10, float4 *h20, float *g0, float4 *psi0, float4 *mu0, int iz, int igpu, cudaStream_t s)
{
    float rho = 0.5;
    cg(ft0, ftn0, h10, h20, g0, psi0, mu0, rho, iz, 1, igpu, s);         
    // gradient(h20, ftn0, 1, iz, igpu, s); 
    // solve_reg(psi0, h20, mu0, lambda0, rho, s);
    // updatemu(mu0, h20, psi0, rho, s);
}
