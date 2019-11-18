
#include "rectv.cuh"
#include "kernels.cuh"
#include <stdio.h>

void rectv::radonapr(float *g, float *f, float tau, int igpu, cudaStream_t s)
{
    //tmp arrays on gpus
    float2 *ftmp0 = (float2 *)ftmp[igpu];
    float2 *ftmps0 = (float2 *)ftmps[igpu];
    float2 *gtmp0 = (float2 *)gtmp[igpu];
    float2 *phi0 = (float2 *)phi[igpu];
    float *theta0 = (float *)theta[igpu];

    cudaMemsetAsync(ftmp0, 0, 2 * N * N * M * Nzp * sizeof(float), s);
    cudaMemsetAsync(ftmps0, 0, 2 * N * N * Nzp * sizeof(float), s);
    cudaMemsetAsync(gtmp0, 0, 2 * N * Ntheta * Nzp * sizeof(float), s);

    //switch to complex numbers
    makecomplex<<<GS3d0, BS3d, 0, s>>>(ftmp0, f, N, N, M * Nzp);
    cudaMemset((void **)&gtmp0, 0, 2 * N * Ntheta * Nzp * sizeof(float));
    for (int i = 0; i < M; i++)
    {
        //decompositon coefficients
        decphi<<<GS3d1, BS3d, 0, s>>>(ftmps0, ftmp0, &phi0[i * Ntheta], N, Ntheta, M, Nzp);
        rad[igpu]->fwdR(gtmp0, ftmps0, theta0, s);
        //multiplication by basis functions
        mulphi<<<GS3d2, BS3d, 0, s>>>(gtmp0, &phi0[i * Ntheta], 1, M, N, Ntheta, Nzp);
        //sum up
        addreal<<<GS3d2, BS3d, 0, s>>>(g, gtmp0, tau, N, Ntheta, Nzp);
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

    cudaMemsetAsync(ftmp0, 0, 2 * N * N * M * Nzp * sizeof(float), s);
    cudaMemsetAsync(ftmps0, 0, 2 * N * N * Nzp * sizeof(float), s);
    cudaMemsetAsync(gtmp0, 0, 2 * N * Ntheta * Nzp * sizeof(float), s);

    for (int i = 0; i < M; i++)
    {
        //switch to complex numbers
        makecomplex<<<GS3d2, BS3d, 0, s>>>(gtmp0, g, N, Ntheta, Nzp);
        //multiplication by conjugate basis functions
        mulphi<<<GS3d2, BS3d, 0, s>>>(gtmp0, &phi0[i * Ntheta], -1, M, N, Ntheta, Nzp); //-1 conj       
        rad[igpu]->adjR(ftmps0, gtmp0, theta0, 0, s);
        //recovering by coefficients
        recphi<<<GS3d1, BS3d, 0, s>>>(ftmp0, ftmps0, &phi0[i * Ntheta], N, Ntheta, M, Nzp);
    }
    addreal<<<GS3d0, BS3d, 0, s>>>(f, ftmp0, -tau, N, N, M * Nzp);
}

void rectv::gradient(float4 *h2, float *ft, float tau, int iz, int igpu, cudaStream_t s)
{
    float *ftmp0 = ftmp[igpu];
    //repeat border values
    extendf<<<GS3d3, BS3d, 0, s>>>(ftmp0, ft, iz != 0, iz != Nz / Nzp - 1, N + 2, M + 2, Nzp + 2);
    grad<<<GS3d3, BS3d, 0, s>>>(h2, ftmp0, tau, lambda1, N + 1, M + 1, Nzp + 1);
}

void rectv::divergent(float *fn, float *f, float4 *h2, float tau, int igpu, cudaStream_t s)
{
    div<<<GS3d0, BS3d, 0, s>>>(fn, f, h2, tau, lambda1, N, M, Nzp);
}

void rectv::adjoint_tests(float *g_, float* theta_, float* phi_)
{
    // only for 1 gpu and 1 slice set    
    // Rapr
    cudaMemcpy(g, g_, N * Ntheta * Nz * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(theta[0], theta_, Ntheta * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(phi[0], phi_, 2 * Ntheta * M * sizeof(float), cudaMemcpyDefault);

    cudaMemset(f, 0, N * N * Nz * M * sizeof(float));
    cudaMemset(fn, 0, N * N * Nz * M * sizeof(float));
    radonapradj(f, g, 1, 0, 0);
    radonapr(h1, f, 1, 0, 0);
    
    double sum[] = {0,0,0};
    float *ftmp = new float[N * N * Nz * M];
    float *h1tmp = new float[N * Ntheta * Nz];
    cudaMemcpy(ftmp, f, N * N * Nz * M * sizeof(float), cudaMemcpyDefault);    
    cudaMemcpy(h1tmp, h1, N * Ntheta * Nz * sizeof(float), cudaMemcpyDefault);
    for (int k = 0; k < N * N * Nz * M; k++) sum[0] += ftmp[k] * ftmp[k];
    for (int k = 0; k < N * Ntheta * Nz; k++) sum[1] += g_[k] * h1tmp[k];
    for (int k = 0; k < N * Ntheta * Nz; k++) sum[2] += h1tmp[k] * h1tmp[k];
    printf("Adjoint test for Rapr: %f ? %f\n", sum[0], -sum[1]);
    printf("Normalization test for Rapr: %f ? %f\n", -sum[1], sum[2]);
    
    // gradient
    gradient(h2, f, 1, 0, 0, 0);
    divergent(fn, fn, h2, 1, 0, 0);
    float *fntmp = new float[N * N * Nz * M];
    float *h2tmp = new float[(N + 1) * (N + 1) * (M + 1) * (Nz + 1)  * sizeof(float4)];
    cudaMemcpy(fntmp, fn, N * N * Nz * M * sizeof(float), cudaMemcpyDefault);    
    cudaMemcpy(h2tmp, h1, (N + 1) * (N + 1) * (M + 1) * (Nz + 1) *  sizeof(float4), cudaMemcpyDefault);
    for (int k = 0; k < N * N * Nz * M; k++) sum[0] += ftmp[k] * fntmp[k];
    for (int k = 0; k < (N + 1) * (N + 1) * (M + 1) * (Nz + 1); k++) sum[1] += h2tmp[k] * h2tmp[k];
    for (int k = 0; k < N * N * Nz * M; k++) sum[2] += fntmp[k] * fntmp[k];
    printf("Adjoint test for grad: %f ? %f\n", sum[0], -sum[1]);
    printf("Normalization test for grad: %f ? %f\n", -sum[1], sum[2]);
    
}

void rectv::adjoint_tests_wrap(float *g, int N1,float *theta, int N2, float *phi, int N3)
{
    adjoint_tests(g, theta, phi);
}