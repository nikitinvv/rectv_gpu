
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

void rectv::cg(float *f0, float *fn0, float *h10, float4 *h20, float *g0, float4 *psi0, float4 *mu0, float rho, int iz, int igpu, cudaStream_t s)       
{
    cudaMemsetAsync(h10, 0, N * Ntheta * Nzp * sizeof(float), s);
    cudaMemsetAsync(h20, 0, (N + 1) * (N + 1) * (M + 1) * (Nzp + 1) * sizeof(float4), s);
    cudaMemcpyAsync(fn0, f0, N * N * Nzp * M * sizeof(float), cudaMemcpyDefault, s);
    //forward step
    gradient(h20, f0, 1, iz, igpu, s); //iz for border control
    radonapr(h10, f0, 1, igpu, s);
    //differences
    diffgrad<<<GS3d4, BS3d, 0, s>>>(h20, psi0, mu0, rho, (N + 1), (M + 1) * (Nzp + 1));
    diff<<<GS3d2, BS3d, 0, s>>>(h10, g0, N, Ntheta, Nzp);
    divergent(fn0, f0, h20, 0.5, igpu, s);
    //backward step
    radonapradj(fn0, h10, 0.5, igpu, s);    
}

void rectv::solver_admm(float *f0, float *fn0, float *h10, float4 *h20, float *g0, float4 *psi0, float4 *mu0, int iz, int igpu, cudaStream_t s)
{
    float rho = 0.5;
    cg(f0, fn0, h10, h20, g0, psi0, mu0, rho, iz, igpu, s);     
}
