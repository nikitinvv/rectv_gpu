#include <stdio.h>
#include <omp.h>
#include "rectv.cuh"

rectv::rectv(size_t n_, size_t ntheta_, size_t M_, size_t nz_, size_t nzp_, size_t ngpus_, float center, float lambda0_, float lambda1_)
{
	n = n_;
	ntheta = ntheta_;
	m = M_;
	nz = nz_;
	nzp = nzp_;
	lambda0 = lambda0_;
	lambda1 = lambda1_;
	
	ngpus = min(ngpus_, (size_t)(nz / nzp));
	omp_set_num_threads(ngpus);
	//Managed memory on GPU
	cudaMallocManaged((void **)&ft, n * n * m * nz * sizeof(float));
	cudaMallocManaged((void **)&ftn, n * n * m * nz * sizeof(float));
	cudaMallocManaged((void **)&g, n * ntheta * nz * sizeof(float));
	cudaMallocManaged((void **)&psi, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * nz / nzp * sizeof(float4));
	cudaMallocManaged((void **)&mu, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * nz / nzp * sizeof(float4));

	//Class for applying Radon transform
	rad = new radonusfft *[ngpus];
	//tmp arrays
	ftmp = new float *[ngpus];
	gtmp = new float *[ngpus];
	ftmps = new float *[ngpus];
	fe = new float *[ngpus];
	h1 = new float *[ngpus];
	h2 = new float4 *[ngpus];
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
		rad[igpu] = new radonusfft(n, ntheta, nzp, center);
		cudaMalloc((void **)&ftmp[igpu], 2 * (n + 2) * (n + 2) * (m + 2) * (nzp + 2) * sizeof(float));
		cudaMalloc((void **)&gtmp[igpu], 2 * n * ntheta * nzp * sizeof(float));
		cudaMalloc((void **)&ftmps[igpu], 2 * n * n * nzp * sizeof(float));
		cudaMalloc((void **)&h1[igpu], n * ntheta * nzp * sizeof(float));
		cudaMalloc((void **)&h2[igpu], (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4));		
		cudaMalloc((void **)&phi[igpu], 2 * ntheta * m * sizeof(float));
		cudaMalloc((void **)&fe[igpu], n * n * m * (nzp + 2) * sizeof(float));
		cudaMalloc((void **)&theta[igpu], ntheta * sizeof(float));		
	}
	cudaDeviceSynchronize();
}

rectv::~rectv()
{
	cudaFree(ft);
	cudaFree(ftn);
	cudaFree(fe);
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
		cudaFree(fe[igpu]);
		cudaFree(h1[igpu]);
		cudaFree(h2[igpu]);		
		cudaFree(phi[igpu]);
		cudaFree(theta[igpu]);
		cudaDeviceReset();
	}
}

void rectv::run(float *fres, float *g_, float *theta_, float *phi_, size_t niter, size_t titer)
{
	//data
	cudaMemcpy(g, g_, n * ntheta * nz * sizeof(float), cudaMemcpyHostToHost);
	//angles and basis functions to each gpu
	for (int igpu = 0; igpu < ngpus; igpu++)
	{
		cudaSetDevice(igpu);
		cudaMemcpy(theta[igpu], theta_, ntheta * sizeof(float), cudaMemcpyDefault);
		cudaMemcpy(phi[igpu], phi_, 2 * ntheta * m * sizeof(float), cudaMemcpyDefault);
	}
	//initial guess
	memset(ft, 0, n * n * m * nz * sizeof(float));
	memset(ftn, 0, n * n * m * nz * sizeof(float));
	memset(psi, 0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * nz / nzp * sizeof(float4));
	memset(mu, 0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * nz / nzp * sizeof(float4));

	float start = omp_get_wtime();
	
#pragma omp parallel
	{
		int igpu = omp_get_thread_num();
		float* h10 = h1[igpu];
    	float4* h20 = h2[igpu];

		printf("gpu: %d\n",igpu);
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
			float *ft0 = &ft[n * n * m * iz * nzp];
			float *ftn0 = &ftn[n * n * m * iz * nzp];
			float4 *psi0 = &psi[(n + 1) * (n + 1) * (m + 1) * iz * (nzp + 1)];
			float4 *mu0 = &mu[(n + 1) * (n + 1) * (m + 1) * iz * (nzp + 1)];
			float *g0 = &g[n * ntheta * iz * nzp];
			cudaMemPrefetchAsync(&ft0[-(iz != 0) * n * n * m], n * n * m * (nzp + 2 - (iz == 0) - (iz == nz / nzp - 1)) * sizeof(float), igpu, s2); //mem+=n*n*m*(nzp+2-(iz==0)-(iz==nz/nzp-1))*sizeof(float);
			cudaMemPrefetchAsync(ftn0, n * n * m * nzp * sizeof(float), igpu, s2);																	//mem+=n*n*m*nzp*sizeof(float);
			cudaMemPrefetchAsync(psi0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), igpu, s2);											//mem+=(n+1)*(n+1)*(m+1)*(nzp+1)*sizeof(float4);
			cudaMemPrefetchAsync(mu0, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), igpu, s2);											//mem+=(n+1)*(n+1)*(m+1)*(nzp+1)*sizeof(float4);
			cudaMemPrefetchAsync(g0, n * ntheta * nzp * sizeof(float), igpu, s2);																	//mem+= n*ntheta*nzp*sizeof(float);
			
			cudaEventRecord(e1, s2);
			float *ft0s = ft0;
			float *ftn0s = ftn0;
			float4 *psi0s = psi0;
			float4 *mu0s = mu0;
			float *g0s = g0;
#pragma omp for
			for (int iz = 0; iz < nz / nzp; iz++)
			{
				cudaEventSynchronize(e1);
				cudaEventSynchronize(e2);
				solver_admm(ft0, ftn0, h10, h20, g0, psi0, mu0, iz, titer, igpu, s1);

				cudaEventRecord(e1, s1);
				if (iz < (igpu + 1) * nz / nzp / ngpus - 1)
				{
					// make sure the stream is idle to force non-deferred HtoD prefetches first
					cudaStreamSynchronize(s2);
					//parts in z
					ft0s = &ft[n * n * m * (iz + 1) * nzp];
					ftn0s = &ftn[n * n * m * (iz + 1) * nzp];
					psi0s = &psi[(n + 1) * (n + 1) * (m + 1) * (iz + 1) * (nzp + 1)];
					mu0s = &mu[(n + 1) * (n + 1) * (m + 1) * (iz + 1) * (nzp + 1)];
					g0s = &g[n * ntheta * (iz + 1) * nzp];
					cudaMemPrefetchAsync(&ft0s[n * n * m], n * n * m * (nzp - (iz + 1 == nz / nzp - 1)) * sizeof(float), igpu, s2); //mem+=n*n*m*(nzp-(iz+1==nz/nzp-1))*sizeof(float);
					cudaMemPrefetchAsync(ftn0s, n * n * m * nzp * sizeof(float), igpu, s2);											//mem+=n*n*m*nzp*sizeof(float);
					cudaMemPrefetchAsync(psi0s, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), igpu, s2);					//mem+=(n+1)*(n+1)*(m+1)*(nzp+1)*sizeof(float4);
					cudaMemPrefetchAsync(mu0s, (n + 1) * (n + 1) * (m + 1) * (nzp + 1) * sizeof(float4), igpu, s2);					//mem+=(n+1)*(n+1)*(m+1)*(nzp+1)*sizeof(float4);
					cudaMemPrefetchAsync(g0s, n * ntheta * nzp * sizeof(float), igpu, s2);											//mem+=n*ntheta*nzp*sizeof(float);

					cudaEventRecord(e2, s2);
				}

				cudaMemPrefetchAsync(&ft0[-(iz != 0) * n * n * m], n * n * m * (nzp - (iz == 0) - (iz == nz / nzp - 1) + 2 * (iz == (igpu + 1) * nz / nzp / ngpus - 1)) * sizeof(float), cudaCpuDeviceId, s1); //mem+= n*n*m*(nzp-(iz==0)-(iz==nz/nzp-1)+2*(iz==(igpu+1)*nz/nzp/ngpus-1))*sizeof(float);
				cudaMemPrefetchAsync(ftn0, n * n * m * nzp * sizeof(float), cudaCpuDeviceId, s1);						  //mem+=n*n*m*nzp*sizeof(float);
				cudaMemPrefetchAsync(g0, n * ntheta * nzp * sizeof(float), cudaCpuDeviceId, s1);						  //mem+=n*ntheta*nzp*sizeof(float);

				ft0 = ft0s;
				ftn0 = ftn0s;
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
				float *tmp = 0;				
				tmp = ft;
				ft = ftn;
				ftn = tmp;
				
				double norm[2] = {};
				for (int k = 0; k < n * n * m * nz; k++)
					norm[0] += (ftn[k] - ft[k]) * (ftn[k] - ft[k]);
				
				// for (int k = 0; k < n * ntheta * nz; k++)
				// 	norm[0] += (h1[k] - g[k]) * (h1[k] - g[k]);
				// for (int kk=0;kk<nz / nzp;kk++)	
				// 	for (int k = 0; k < (n + 1) * (n + 1) * (m + 1) * (nzp); k++)
				// 	{
				// 		int id = kk* (n + 1) * (n + 1) * (m + 1) * (nzp + 1)+k;
				// 		norm[1] += sqrt(h2[id].x * h2[id].x + h2[id].y * h2[id].y + h2[id].z * h2[id].z + h2[id].w * h2[id].w);
				// 	}	
				fprintf(stderr, "iterations (%d/%d) f:%f, r:%f, total:%f\n", iter, niter, norm[0], lambda0 * norm[1], norm[0] + lambda0 * norm[1]);
				fflush(stdout);
			}
		}
		cudaDeviceSynchronize();
#pragma omp barrier
	}
	float end = omp_get_wtime();
	printf("Elapsed time: %fs.\n", end - start);
	cudaMemPrefetchAsync(ft, n * n * m * nz * sizeof(float), cudaCpuDeviceId, 0);
	cudaMemcpy(fres, ft, n * n * m * nz * sizeof(float), cudaMemcpyDefault);
}

void rectv::run_wrap(float *fres, int n0, float *g, int n1, float *theta, int n2, float *phi, int n3, size_t niter, size_t titer)
{
	run(fres, g, theta, phi, niter, titer);
}