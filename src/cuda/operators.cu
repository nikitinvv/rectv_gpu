
#include "rectv.cuh"
#include "kernels_operators.cuh"
#include <stdio.h>

void rectv::radonapr(float *g, float *f, float tau, int igpu, cudaStream_t s)
{
    //tmp arrays on gpus
    float2 *ftmp0 = (float2 *)ftmp[igpu];
    float2 *ftmps0 = (float2 *)ftmps[igpu];
    float2 *gtmp0 = (float2 *)gtmp[igpu];
    float2 *phi0 = (float2 *)phi[igpu];
    float *theta0 = (float *)theta[igpu];

    cudaMemsetAsync(ftmp0, 0, 2 * n * n * m * nzp * sizeof(float), s);
    cudaMemsetAsync(ftmps0, 0, 2 * n * n * nzp * sizeof(float), s);
    cudaMemsetAsync(gtmp0, 0, 2 * n * ntheta * nzp * sizeof(float), s);
    cudaMemsetAsync(g, 0, n * ntheta * nzp * sizeof(float), s);

    //switch to complex numbers
    makecomplex<<<GS3d0, BS3d, 0, s>>>(ftmp0, f, n, n, m * nzp);
    for (int i = 0; i < m; i++)
    {
        //decompositon coefficients
        decphi<<<GS3d1, BS3d, 0, s>>>(ftmps0, ftmp0, &phi0[i * ntheta], n, ntheta, m, nzp);
        rad[igpu]->fwdR(gtmp0, ftmps0, theta0, s);
        //multiplication by basis functions
        mulphi<<<GS3d2, BS3d, 0, s>>>(gtmp0, &phi0[i * ntheta], 1, m, n, ntheta, nzp);
        //sum up
        addreal<<<GS3d2, BS3d, 0, s>>>(g, gtmp0, tau, n, ntheta, nzp);
    }
}

void rectv::radonapradj(float *f, float *g, float tau, int igpu, cudaStream_t s)
{
    //tmp arrays on gpus
    float2 *ftmp0 = (float2 *)ftmp[igpu];
    float2 *ftmps0 = (float2 *)ftmps[igpu];
    float2 *gtmp0 = (float2 *)gtmp[igpu];
    float2 *phi0 = (float2 *)phi[igpu];
    float *theta0 = (float *)theta[igpu];

    cudaMemsetAsync(ftmp0, 0, 2 * n * n * m * nzp * sizeof(float), s);
    cudaMemsetAsync(ftmps0, 0, 2 * n * n * nzp * sizeof(float), s);
    cudaMemsetAsync(gtmp0, 0, 2 * n * ntheta * nzp * sizeof(float), s);

    for (int i = 0; i < m; i++)
    {
        //switch to complex numbers
        makecomplex<<<GS3d2, BS3d, 0, s>>>(gtmp0, g, n, ntheta, nzp);
        //multiplication by conjugate basis functions
        mulphi<<<GS3d2, BS3d, 0, s>>>(gtmp0, &phi0[i * ntheta], -1, m, n, ntheta, nzp); //-1 conj       
        rad[igpu]->adjR(ftmps0, gtmp0, theta0, 0, s);
        //recovering by coefficients
        recphi<<<GS3d1, BS3d, 0, s>>>(ftmp0, ftmps0, &phi0[i * ntheta], n, ntheta, m, nzp);
    }
    addreal<<<GS3d0, BS3d, 0, s>>>(f, ftmp0, tau, n, n, m * nzp);
}

void rectv::gradient(float4 *h2, float *f, float lambda1, int iz, int igpu, cudaStream_t s)
{
    float *ftmp0 = ftmp[igpu];
    //repeat border values
    extendf<<<GS3d3, BS3d, 0, s>>>(ftmp0, f, iz != 0, iz != nz / nzp - 1, n + 2, m + 2, nzp + 2);
    gradf<<<GS3d3, BS3d, 0, s>>>(h2, ftmp0, lambda1, n + 1, m + 1, nzp + 1);
}

void rectv::divergent(float *fn, float4 *h2, float lambda1, float tau, int igpu, cudaStream_t s)
{
    div<<<GS3d0, BS3d, 0, s>>>(fn, h2, tau, lambda1, n, m, nzp);
}
