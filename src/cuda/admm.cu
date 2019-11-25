
#include "rectv.cuh"
#include "stdio.h"

__global__ void diff(float *h1, float *g, int n, int ntheta, int nz)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= n || ty >= ntheta || tz >= nz)
        return;

    int id0 = tx + ty * n + tz * n * ntheta;
    h1[id0] = (h1[id0] - g[id0]);
}

__global__ void updatemu_ker(float4 *mu, float4 *h2, float4 *psi, float rho, int n, int nz)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= n || ty >= n || tz >= nz)
        return;

    int id0 = tx + ty * n + tz * n * n;
    mu[id0].x += rho * (h2[id0].x - psi[id0].x);
    mu[id0].y += rho * (h2[id0].y - psi[id0].y);
    mu[id0].z += rho * (h2[id0].z - psi[id0].z);
    mu[id0].w += rho * (h2[id0].w - psi[id0].w);
}

__global__ void diffgrad(float4 *h2, float4 *psi, float4 *mu, float rho, int n, int nz)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= n || ty >= n || tz >= nz)
        return;

    int id0 = tx + ty * n + tz * n * n;
    h2[id0].x = rho * (h2[id0].x - psi[id0].x + mu[id0].x / rho);
    h2[id0].y = rho * (h2[id0].y - psi[id0].y + mu[id0].y / rho);
    h2[id0].z = rho * (h2[id0].z - psi[id0].z + mu[id0].z / rho);
    h2[id0].w = rho * (h2[id0].w - psi[id0].w + mu[id0].w / rho);
}

__global__ void solve_reg_ker(float4* psi, float4 *h2, float4* mu, float lambda, float rho, int n, int nz)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= n || ty >= n || tz >= nz)
        return;

    int id0 = tx + ty * n + tz * n * n;
    psi[id0].x =(h2[id0].x+mu[id0].x/rho);
    psi[id0].y =(h2[id0].y+mu[id0].y/rho);
    psi[id0].z =(h2[id0].z+mu[id0].z/rho);
    psi[id0].w =(h2[id0].w+mu[id0].w/rho);    
	float za = sqrtf(psi[id0].x * psi[id0].x + 
                     psi[id0].y * psi[id0].y + 
                     psi[id0].z * psi[id0].z + 
                     psi[id0].w * psi[id0].w);
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

void rectv::solver_admm(float *f, float *fn, float* h1, float4* h2, float* fm, float *g, float4 *psi, float4 *mu, int iz, int titer, int igpu, cudaStream_t s)
{
    float rho = 0.5;
    
    for (int k=0;k<titer;k++)
    {
        //forward step
        gradient(h2, fm, 1, iz, igpu, s); //iz for border control
        radonapr(h1, fm, 1, igpu, s);        
        //differences
        diffgrad<<<GS3d4, BS3d, 0, s>>>(h2, psi, mu, rho, (n + 1), (m + 1) * (nzp + 1));
        diff<<<GS3d2, BS3d, 0, s>>>(h1, g, n, ntheta, nzp);
        divergent(fm, fm, h2, 0.5/lambda1, igpu, s);
        //backward step
        radonapradj(fm, h1, 0.5/lambda1, igpu, s);         
    }    
    cudaMemsetAsync(h2, 0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), s);        
    //forward step
    gradient(h2, fm, 1, iz, igpu, s); //iz for border control
    cudaMemcpyAsync(fn, fm, n * n * nzp * m * sizeof(float), cudaMemcpyDefault, s);
    // solve reg
    solve_reg_ker<<<GS3d4, BS3d, 0, s>>>(psi, h2, mu, lambda0, rho, n+1, (m+1)*(nzp+1));   
    updatemu_ker<<<GS3d4, BS3d, 0, s>>>(mu, h2, psi, rho, n+1, (m+1)*(nzp+1));   
    // // cudaDeviceSynchronize();
    // double norm=0;
    // for( int id=0;id< (n + 1)*(n + 1) * (m + 1) * (nzp);id++) 
    //     norm+=sqrt(h20[id].x * h20[id].x + h20[id].y * h20[id].y + h20[id].z * h20[id].z + h20[id].w * h20[id].w);

    // printf(" %f\n",lambda0*norm);

}
