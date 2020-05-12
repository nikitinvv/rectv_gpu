#include <stdio.h>
#include <omp.h>
#include "rectv.cuh"

rectv::rectv(int n, int ntheta, int m, int nz, int nzp_, int ngpus_)
 : n(n), ntheta(ntheta), m(m), nz(nz) {

	nzp = nzp_;
	ngpus = min(ngpus_, (int)(nz / nzp));
	omp_set_num_threads(ngpus);
	//Managed memory on GPU
	cudaMallocManaged((void **)&f, n * n * m * nz * sizeof(float));
	cudaMallocManaged((void **)&fn, n * n * m * nz * sizeof(float));
	cudaMallocManaged((void **)&g, n * ntheta * nz * sizeof(float));
	cudaMallocManaged((void **)&psi, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * nz / nzp * sizeof(float4));
	cudaMallocManaged((void **)&mu, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * nz / nzp * sizeof(float4));

	//Class for applying Radon transform
	rad = new radonusfft *[ngpus];
	cublas_handles = new cublasHandle_t[ngpus];
	//tmp arrays
	ftmp = new float *[ngpus];
	gtmp = new float *[ngpus];
	ftmps = new float *[ngpus];
	fm = new float *[ngpus];
	h1 = new float *[ngpus];
	h2 = new float4 *[ngpus];
	h2stored = new float4 *[ngpus];
	phi = new float2 *[ngpus];
	theta = new float *[ngpus];
	
	BS2d = dim3(32, 32);
	BS3d = dim3(32, 32, 1);
	GS2d0 = dim3(ceil(ntheta / (float)BS2d.x), ceil(m / (float)BS2d.y));

	GS3d0 = dim3(ceil(n / (float)BS3d.x), ceil(n / (float)BS3d.y), ceil(nzp * m / (float)BS3d.z));
	GS3d1 = dim3(ceil(n / (float)BS3d.x), ceil(n / (float)BS3d.y), ceil(nzp / (float)BS3d.z));
	GS3d2 = dim3(ceil(n / (float)BS3d.x), ceil(ntheta / (float)BS3d.y), ceil(nzp / (float)BS3d.z));
	GS3d3 = dim3(ceil((n + 2) / (float)BS3d.x), ceil((n + 2) / (float)BS3d.y), ceil((m + 2) * (nzp + 2) / (float)BS3d.z));
	GS3d4 = dim3(ceil((n + 1) / (float)BS3d.x), ceil((n + 1) / (float)BS3d.y), ceil((m + 1) * (nzp + 1) / (float)BS3d.z));

	for (int igpu = 0; igpu < ngpus; igpu++)
	{
		cudaSetDevice(igpu);
		rad[igpu] = new radonusfft(n, ntheta, nzp);
		cudaMalloc((void **)&ftmp[igpu], 2 * (n + 2) * (n + 2) * (m + 2) * (nzp + 2) * sizeof(float));
		cudaMalloc((void **)&gtmp[igpu], 2 * n * ntheta * nzp * sizeof(float));
		cudaMalloc((void **)&ftmps[igpu], 2 * n * n * nzp * sizeof(float));
		cudaMalloc((void **)&h1[igpu], n * ntheta * nzp * sizeof(float));
		cudaMalloc((void **)&h2[igpu], (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4));		
		cudaMalloc((void **)&h2stored[igpu], (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4));				
		cudaMalloc((void **)&phi[igpu], 2 * ntheta * m * sizeof(float));
		cudaMalloc((void **)&fm[igpu], n * n * m * (nzp + 2) * sizeof(float));
		cudaMalloc((void **)&theta[igpu], ntheta * sizeof(float));		
		cublasCreate_v2(&cublas_handles[igpu]);
	}
	cudaDeviceSynchronize();
	is_free = false;
}

rectv::~rectv() { free(); }

void rectv::free()
{
	if (!is_free) 
	{
		cudaFree(f);
		cudaFree(fn);
		cudaFree(fm);
		cudaFree(g);
		cudaFree(psi);
		cudaFree(mu);
		for (int igpu = 0; igpu < ngpus; igpu++)
		{
			cudaSetDevice(igpu);
			delete rad[igpu];
			cudaFree(ftmp[igpu]);
			cudaFree(gtmp[igpu]);
			cudaFree(ftmps[igpu]);
			cudaFree(fm[igpu]);
			cudaFree(h1[igpu]);
			cudaFree(h2[igpu]);		
			cudaFree(h2stored[igpu]);		
			cudaFree(phi[igpu]);
			cudaFree(theta[igpu]);
			cublasDestroy(cublas_handles[igpu]);
			cudaDeviceReset();
		}
		is_free = true;
	}
}

void rectv::set_center(float center)
{
	for (int igpu = 0; igpu < ngpus; igpu++)
	{
		cudaSetDevice(igpu);
		rad[igpu]->set_center(center);
	}
}

void rectv::run(size_t fres, size_t g_, size_t theta_, size_t phi_, 
	float center, float lambda0, float lambda1, 
	int niter, int titer)
{
	//update center with a given one
	set_center(center);
	
	//copy data
	cudaMemcpy(g, (float*)g_, n * ntheta * nz * sizeof(float), cudaMemcpyHostToHost);
	//angles and basis functions to each gpu
	for (int igpu = 0; igpu < ngpus; igpu++)
	{
		cudaSetDevice(igpu);
		cudaMemcpy(theta[igpu], (float*)theta_, ntheta * sizeof(float), cudaMemcpyDefault);
		cudaMemcpy(phi[igpu], (float*)phi_, 2 * ntheta * m * sizeof(float), cudaMemcpyDefault);
	}
	//initial guess
	memset(f, 0, n * n * m * nz * sizeof(float));
	memset(fn, 0, n * n * m * nz * sizeof(float));
	memset(psi, 0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * nz / nzp * sizeof(float4));
	memset(mu, 0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * nz / nzp * sizeof(float4));

	//weighting factor for the ADMM scheme
	float rho = 0.5;
	//norms for updating the weighting factor on each iteration
	float2* normdiff = new float2[nz/nzp];
#pragma omp parallel
	{
		int igpu = omp_get_thread_num();
	
		cudaSetDevice(igpu);
		cudaStream_t s1, s2, s3, st;
		cudaEvent_t e1, e2, et;
		cudaStreamCreate(&s1);
		cudaStreamCreate(&s2);
		cudaStreamCreate(&s3);
		cudaEventCreate(&e1);
		cudaEventCreate(&e2);
		for (int iter = 0; iter < niter; iter++)
		{
			//parts in z
			int iz = igpu * nz / nzp / ngpus;
			float *f0 = &f[n * n * m * iz * nzp];
			float *fn0 = &fn[n * n * m * iz * nzp];
			float4 *psi0 = &psi[(n + 1) * (n + 1) * (m + 1) * iz * (nzp + 1)];
			float4 *mu0 = &mu[(n + 1) * (n + 1) * (m + 1) * iz * (nzp + 1)];
			float *g0 = &g[n * ntheta * iz * nzp];
			cudaMemPrefetchAsync(&f0[-(iz != 0) * n * n * m], n * n * m * (nzp + 2 - (iz == 0) - (iz == nz / nzp - 1)) * sizeof(float), igpu, s2); //mem+=n*n*m*(nzp+2-(iz==0)-(iz==nz/nzp-1))*sizeof(float);
			cudaMemPrefetchAsync(fn0, n * n * m * nzp * sizeof(float), igpu, s2);																	//mem+=n*n*m*nzp*sizeof(float);
			cudaMemPrefetchAsync(psi0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), igpu, s2);											//mem+=(n+1)*(n+1)*(m+1)*(nzp+1)*sizeof(float4);
			cudaMemPrefetchAsync(mu0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), igpu, s2);											//mem+=(n+1)*(n+1)*(m+1)*(nzp+1)*sizeof(float4);
			cudaMemPrefetchAsync(g0, n * ntheta * nzp * sizeof(float), igpu, s2);																	//mem+= n*ntheta*nzp*sizeof(float);
			
			cudaEventRecord(e1, s2);
			float *f0s = f0;
			float *fn0s = fn0;
			float4 *psi0s = psi0;
			float4 *mu0s = mu0;
			float *g0s = g0;
#pragma omp for
			for (int iz = 0; iz < nz / nzp; iz++)
			{
				cudaEventSynchronize(e1);
				cudaEventSynchronize(e2);
				// intermediate arrays
				float* h10 = h1[igpu];
    			float4* h20 = h2[igpu];
				float4* h2stored0 = h2stored[igpu];
				float* fm0 = &fm[igpu][(iz != 0) * n * n * m];//modifyable version of f
    			cudaMemcpyAsync(&fm0[-(iz != 0) * n * n * m],&f0[-(iz != 0) * n * n * m],n*n*(nzp + 2 - (iz == 0) - (iz == nz / nzp - 1))*m* sizeof(float), cudaMemcpyDefault, s1); //mem+=n*n*m*(nzp+2-(iz==0)-(iz==nz/nzp-1))*sizeof(float);
				// ADMM
				normdiff[iz] = solver_admm(f0, fn0, h10, h20, h2stored0, fm0, g0, psi0, mu0, lambda0, lambda1, rho, iz, titer, igpu, s1);

				cudaEventRecord(e1, s1);
				if (iz < (igpu + 1) * nz / nzp / ngpus - 1)
				{
					// make sure the stream is idle to force non-deferred HtoD prefetches first
					cudaStreamSynchronize(s2);
					//parts in z
					f0s = &f[n * n * m * (iz + 1) * nzp];
					fn0s = &fn[n * n * m * (iz + 1) * nzp];
					psi0s = &psi[(n + 1) * (n + 1) * (m + 1) * (iz + 1) * (nzp + 1)];
					mu0s = &mu[(n + 1) * (n + 1) * (m + 1) * (iz + 1) * (nzp + 1)];
					g0s = &g[n * ntheta * (iz + 1) * nzp];
					cudaMemPrefetchAsync(&f0s[n * n * m], n * n * m * (nzp - (iz + 1 == nz / nzp - 1)) * sizeof(float), igpu, s2); //mem+=n*n*m*(nzp-(iz+1==nz/nzp-1))*sizeof(float);
					cudaMemPrefetchAsync(fn0s, n * n * m * nzp * sizeof(float), igpu, s2);											//mem+=n*n*m*nzp*sizeof(float);
					cudaMemPrefetchAsync(psi0s, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), igpu, s2);					//mem+=(n+1)*(n+1)*(m+1)*(nzp+1)*sizeof(float4);
					cudaMemPrefetchAsync(mu0s, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), igpu, s2);					//mem+=(n+1)*(n+1)*(m+1)*(nzp+1)*sizeof(float4);
					cudaMemPrefetchAsync(g0s, n * ntheta * nzp * sizeof(float), igpu, s2);											//mem+=n*ntheta*nzp*sizeof(float);

					cudaEventRecord(e2, s2);
				}

				cudaMemPrefetchAsync(&f0[-(iz != 0) * n * n * m], n * n * m * (nzp - (iz == 0) - (iz == nz / nzp - 1) + 2 * (iz == (igpu + 1) * nz / nzp / ngpus - 1)) * sizeof(float), cudaCpuDeviceId, s1); //mem+= n*n*m*(nzp-(iz==0)-(iz==nz/nzp-1)+2*(iz==(igpu+1)*nz/nzp/ngpus-1))*sizeof(float);
				cudaMemPrefetchAsync(fn0, n * n * m * nzp * sizeof(float), cudaCpuDeviceId, s1);						  //mem+=n*n*m*nzp*sizeof(float);
				cudaMemPrefetchAsync(g0, n * ntheta * nzp * sizeof(float), cudaCpuDeviceId, s1);						  //mem+=n*ntheta*nzp*sizeof(float);

				f0 = f0s;
				fn0 = fn0s;
				psi0 = psi0s;
				mu0 = mu0s;
				g0 = g0s;
				// rotate streams and swap events
				st = s1;
				s1 = s2;
				s2 = st;
				st = s2;
				s2 = s3;
				s3 = st;
				et = e1;
				e1 = e2;
				e2 = et;
			}

			cudaEventSynchronize(e1);
			cudaEventSynchronize(e2);
			cudaDeviceSynchronize();
#pragma omp barrier
#pragma omp single
			{

				//update rho
				float r=0;
				float s=0;
				for(int k=0;k<nz/nzp;k++)
				{
					r += normdiff[k].x*normdiff[k].x;
					s += rho*rho*normdiff[k].y*normdiff[k].y;
				}
				if(r>10*s) rho*=2;
				else if (s>10*r) rho*=0.5;
				
				//check convergence
				double norm=0;
				for (int k = 0; k < n * n * m * nz; k++)
					norm += (fn[k] - f[k]) * (fn[k] - f[k]);
				fprintf(stderr, "iter (%d/%d), rdiff ||f(k+1)-f(k)||:%f \n", iter, niter, norm);fflush(stdout);
				
				//swap old/new object
				float *tmp = 0;				
				tmp = f;
				f = fn;
				fn = tmp;			
			}
		}
		cudaDeviceSynchronize();
#pragma omp barrier
	}
	cudaMemPrefetchAsync(f, n * n * m * nz * sizeof(float), cudaCpuDeviceId, 0);
	cudaMemcpy((float*)fres, f, n * n * m * nz * sizeof(float), cudaMemcpyDefault);
}


void rectv::check_adjoints(size_t res_, size_t g_, size_t theta_, size_t phi_, float lambda1, float center)
{
	double* res = (double*)res_;
    // only for 1 gpu and 1 slice set    
    //update center with a given one
	set_center(center);
    // Rapr
    cudaMemcpy(g, (float*)g_, n * ntheta * nz * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(theta[0], (float*)theta_, ntheta * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(phi[0], (float*)phi_, 2 * ntheta * m * sizeof(float), cudaMemcpyDefault);

    radonapradj(f, g, 1, 0, 0);
    radonapr(h1[0], f, 1, 0, 0);
    
    float *ftmp = new float[n * n * nz * m];
    float *h1tmp = new float[n * ntheta * nz];
    cudaMemcpy(ftmp, f, n * n * nz * m * sizeof(float), cudaMemcpyDefault);    
    cudaMemcpy(h1tmp, h1[0], n * ntheta * nz * sizeof(float), cudaMemcpyDefault);
    for (int k = 0; k < n * n * nz * m; k++) res[0] += ftmp[k] * ftmp[k];
    for (int k = 0; k < n * ntheta * nz; k++) res[1] += ((float*)g_)[k] * h1tmp[k];
    for (int k = 0; k < n * ntheta * nz; k++) res[2] += h1tmp[k] * h1tmp[k];
    
    // gradient
    gradient(h2[0], f, lambda1, 0, 0, 0);
   
    divergent(fn, h2[0], lambda1, 1,  0, 0);
     
    float *fntmp = new float[n * n * nz * m];
    float *h2tmp = new float[(n + 1) * (n + 1) * (m + 1) * (nz + 1)  * sizeof(float4)];
    cudaMemcpy(fntmp, fn, n * n * nz * m * sizeof(float), cudaMemcpyDefault);    
    cudaMemcpy(h2tmp, h2[0], (n + 1) * (n + 1) * (m + 1) * (nz + 1) *  sizeof(float4), cudaMemcpyDefault);
    for (int k = 0; k < n * n * nz * m; k++) res[3] += ftmp[k] * fntmp[k];
    for (int k = 0; k < (n + 1) * (n + 1) * (m + 1) * (nz + 1); k++) res[4] += h2tmp[k] * h2tmp[k];
    for (int k = 0; k < n * n * nz * m; k++) res[5] += fntmp[k] * fntmp[k];
    
}