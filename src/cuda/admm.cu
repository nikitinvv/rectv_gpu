
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

void rectv::cg(float *ft0, float *ftn0, float *h10, float4 *h20, float *g0, float4 *psi0, float4 *mu0, float rho, int iz, int niter, int igpu, cudaStream_t s)       
{
    float* ft00 = &fe[igpu][(iz != 0) * n * n * m];//modifyable version of ft0
    cudaMemcpyAsync(&ft00[-(iz != 0) * n * n * m],&ft0[-(iz != 0) * n * n * m],n*n*(nzp + 2 - (iz == 0) - (iz == nz / nzp - 1))*m* sizeof(float), cudaMemcpyDefault, s); //mem+=n*n*m*(nzp+2-(iz==0)-(iz==nz/nzp-1))*sizeof(float);
    
    for (int k=0;k<niter;k++)
    {
        cudaMemsetAsync(h10, 0, n * ntheta * nzp * sizeof(float), s);
        cudaMemsetAsync(h20, 0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), s);        
        //forward step
        gradient(h20, ft00, 1, iz, igpu, s); //iz for border control
        radonapr(h10, ft00, 1, igpu, s);        
        //differences
        diffgrad<<<GS3d4, BS3d, 0, s>>>(h20, psi0, mu0, rho, (n + 1), (m + 1) * (nzp + 1));
        diff<<<GS3d2, BS3d, 0, s>>>(h10, g0, n, ntheta, nzp);
        divergent(ft00, ft00, h20, 0.5/lambda1, igpu, s);
        //backward step
        radonapradj(ft00, h10, 0.5/lambda1, igpu, s);         
    }    
    cudaMemsetAsync(h20, 0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), s);        
    //forward step
    gradient(h20, ft00, 1, iz, igpu, s); //iz for border control
    cudaMemcpyAsync(ftn0, ft00, n * n * nzp * m * sizeof(float), cudaMemcpyDefault, s);
}

void rectv::solve_reg(float4* psi, float4* h2, float4* mu, float lambda, float rho, cudaStream_t s)
{
    solve_reg_ker<<<GS3d4, BS3d, 0, s>>>(psi, h2, mu, lambda, rho, n+1, (m+1)*(nzp+1));   
}

void rectv::updatemu(float4* mu, float4* h2, float4* psi, float rho, cudaStream_t s)
{
    updatemu_ker<<<GS3d4, BS3d, 0, s>>>(mu, h2, psi, rho, n+1, (m+1)*(nzp+1));    
}

void rectv::solver_admm(float *ft0, float *ftn0, float *h10, float4 *h20, float *g0, float4 *psi0, float4 *mu0, int iz, int titer, int igpu, cudaStream_t s)
{
    float rho = 0.5;

    cg(ft0, ftn0, h10, h20, g0, psi0, mu0, rho, iz, titer, igpu, s);         
    solve_reg(psi0, h20, mu0, lambda0, rho, s);
    updatemu(mu0, h20, psi0, rho, s);
    // cudaDeviceSynchronize();
    // double norm=0;
    // for( int id=0;id< (n + 1)*(n + 1) * (m + 1) * (nzp);id++) 
    //     norm+=sqrt(h20[id].x * h20[id].x + h20[id].y * h20[id].y + h20[id].z * h20[id].z + h20[id].w * h20[id].w);

    // printf(" %f\n",lambda0*norm);

}
