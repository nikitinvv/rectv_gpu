#include <stdio.h>
#include <omp.h>
#include "rectv.cuh"

rectv::rectv(size_t N_, size_t Ntheta_, size_t M_, size_t Nz_, size_t Nzp_, size_t ngpus_, float center, float lambda0_, float lambda1_)
{
	N = N_;
	Ntheta = Ntheta_;
	M = M_;
	Nz = Nz_;
	Nzp = Nzp_;
	lambda0 = lambda0_;
	lambda1 = lambda1_;
	ngpus = min(ngpus_, (size_t)(Nz / Nzp));
	omp_set_num_threads(ngpus);
	//Managed memory on GPU
	cudaMallocManaged((void **)&f, N * N * M * Nz * sizeof(float));
	cudaMallocManaged((void **)&fn, N * N * M * Nz * sizeof(float));
	cudaMallocManaged((void **)&ft, N * N * M * Nz * sizeof(float));
	cudaMallocManaged((void **)&ftn, N * N * M * Nz * sizeof(float));
	cudaMallocManaged((void **)&g, N * Ntheta * Nz * sizeof(float));
	cudaMallocManaged((void **)&h1, N * Ntheta * Nz * sizeof(float));
	cudaMallocManaged((void **)&h2, (N + 1) * (N + 1) * (M + 1) * (Nzp + 1) * Nz / Nzp * sizeof(float4));

	//Class for applying Radon transform
	rad = new radonusfft *[ngpus];
	//tmp arrays
	ftmp = new float *[ngpus];
	gtmp = new float *[ngpus];
	ftmps = new float *[ngpus];
	phi = new float2 *[ngpus];
	theta = new float *[ngpus];

	BS2d = dim3(32, 32);
	BS3d = dim3(32, 32, 1);
	GS2d0 = dim3(ceil(Ntheta / (float)BS2d.x), ceil(M / (float)BS2d.y));

	GS3d0 = dim3(ceil(N / (float)BS3d.x), ceil(N / (float)BS3d.y), ceil(Nzp * M / (float)BS3d.z));
	GS3d1 = dim3(ceil(N / (float)BS3d.x), ceil(N / (float)BS3d.y), ceil(Nzp / (float)BS3d.z));
	GS3d2 = dim3(ceil(N / (float)BS3d.x), ceil(Ntheta / (float)BS3d.y), ceil(Nzp / (float)BS3d.z));
	GS3d3 = dim3(ceil((N + 2) / (float)BS3d.x), ceil((N + 2) / (float)BS3d.y), ceil((M + 2) * (Nzp + 2) / (float)BS3d.z));
	GS3d4 = dim3(ceil((N + 1) / (float)BS3d.x), ceil((N + 1) / (float)BS3d.y), ceil((M + 1) * (Nzp + 1) / (float)BS3d.z));

	for (int igpu = 0; igpu < ngpus; igpu++)
	{
		cudaSetDevice(igpu);
		rad[igpu] = new radonusfft(N, Ntheta, Nzp, center);
		cudaMalloc((void **)&ftmp[igpu], 2 * (N + 2) * (N + 2) * (M + 2) * (Nzp + 2) * sizeof(float));
		cudaMalloc((void **)&gtmp[igpu], 2 * N * Ntheta * Nzp * sizeof(float));
		cudaMalloc((void **)&ftmps[igpu], 2 * N * N * Nzp * sizeof(float));
		cudaMalloc((void **)&phi[igpu], 2 * Ntheta * M * sizeof(float));
		cudaMalloc((void **)&theta[igpu], Ntheta * sizeof(float));		
	}
	cudaDeviceSynchronize();
}

rectv::~rectv()
{
	cudaFree(f);
	cudaFree(fn);
	cudaFree(ft);
	cudaFree(ftn);
	cudaFree(g);
	cudaFree(h1);
	cudaFree(h2);
	for (int igpu = 0; igpu < ngpus; igpu++)
	{
		cudaSetDevice(igpu);
		delete rad[igpu];
		cudaFree(ftmp[igpu]);
		cudaFree(gtmp[igpu]);
		cudaFree(ftmps[igpu]);
		cudaFree(phi[igpu]);
		cudaFree(theta[igpu]);
		cudaDeviceReset();
	}
}

void rectv::run(float *fres, float *g_, float *theta_, float *phi_, size_t niter)
{
	//data
	cudaMemcpy(g, g_, N * Ntheta * Nz * sizeof(float), cudaMemcpyHostToHost);
	//angles and basis functions to each gpu
	for (int igpu = 0; igpu < ngpus; igpu++)
	{
		cudaSetDevice(igpu);
		cudaMemcpy(theta[igpu], theta_, Ntheta * sizeof(float), cudaMemcpyDefault);
		cudaMemcpy(phi[igpu], phi_, 2 * Ntheta * M * sizeof(float), cudaMemcpyDefault);
	}
	//initial guess
	memset(f, 0, N * N * M * Nz * sizeof(float));
	memset(ft, 0, N * N * M * Nz * sizeof(float));
	memset(fn, 0, N * N * M * Nz * sizeof(float));
	memset(ftn, 0, N * N * M * Nz * sizeof(float));
	memset(h1, 0, N * Ntheta * Nz * sizeof(float));
	memset(h2, 0, (N + 1) * (N + 1) * (M + 1) * (Nzp + 1) * Nz / Nzp * sizeof(float4));

	float start = omp_get_wtime();
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
			int iz = igpu * Nz / Nzp / ngpus;
			float *f0 = &f[N * N * M * iz * Nzp];
			float *fn0 = &fn[N * N * M * iz * Nzp];
			float *ft0 = &ft[N * N * M * iz * Nzp];
			float *ftn0 = &ftn[N * N * M * iz * Nzp];
			float *h10 = &h1[N * Ntheta * iz * Nzp];
			float4 *h20 = &h2[(N + 1) * (N + 1) * (M + 1) * iz * (Nzp + 1)];
			float *g0 = &g[N * Ntheta * iz * Nzp];
			cudaMemPrefetchAsync(f0, N * N * M * Nzp * sizeof(float), igpu, s2);																	//mem+=N*N*M*Nzp*sizeof(float);
			cudaMemPrefetchAsync(fn0, N * N * M * Nzp * sizeof(float), igpu, s2);																	//mem+=N*N*M*Nzp*sizeof(float);
			cudaMemPrefetchAsync(&ft0[-(iz != 0) * N * N * M], N * N * M * (Nzp + 2 - (iz == 0) - (iz == Nz / Nzp - 1)) * sizeof(float), igpu, s2); //mem+=N*N*M*(Nzp+2-(iz==0)-(iz==Nz/Nzp-1))*sizeof(float);
			cudaMemPrefetchAsync(ftn0, N * N * M * Nzp * sizeof(float), igpu, s2);																	//mem+=N*N*M*Nzp*sizeof(float);
			cudaMemPrefetchAsync(h10, N * Ntheta * Nzp * sizeof(float), igpu, s2);																	//mem+=N*Ntheta*Nzp*sizeof(float);
			cudaMemPrefetchAsync(h20, (N + 1) * (N + 1) * (M + 1) * (Nzp + 1) * sizeof(float4), igpu, s2);											//mem+=(N+1)*(N+1)*(M+1)*(Nzp+1)*sizeof(float4);
			cudaMemPrefetchAsync(g0, N * Ntheta * Nzp * sizeof(float), igpu, s2);																	//mem+= N*Ntheta*Nzp*sizeof(float);

			cudaEventRecord(e1, s2);
			float *f0s = f0;
			float *fn0s = fn0;
			float *ft0s = ft0;
			float *ftn0s = ftn0;
			float *h10s = h10;
			float4 *h20s = h20;
			float *g0s = g0;
#pragma omp forrectv
			for (int iz = 0; iz < Nz / Nzp; iz++)
			{
				cudaEventSynchronize(e1);
				cudaEventSynchronize(e2);

				solver_chambolle(f0, fn0, ft0, ftn0, h10, h20, g0, iz, igpu, s1);

				cudaEventRecord(e1, s1);
				if (iz < (igpu + 1) * Nz / Nzp / ngpus - 1)
				{
					// make sure the stream is idle to force non-deferred HtoD prefetches first
					cudaStreamSynchronize(s2);
					//parts in z
					f0s = &f[N * N * M * (iz + 1) * Nzp];
					fn0s = &fn[N * N * M * (iz + 1) * Nzp];
					ft0s = &ft[N * N * M * (iz + 1) * Nzp];
					ftn0s = &ftn[N * N * M * (iz + 1) * Nzp];
					h10s = &h1[N * Ntheta * (iz + 1) * Nzp];
					h20s = &h2[(N + 1) * (N + 1) * (M + 1) * (iz + 1) * (Nzp + 1)];
					g0s = &g[N * Ntheta * (iz + 1) * Nzp];
					cudaMemPrefetchAsync(f0s, N * N * M * Nzp * sizeof(float), igpu, s2);											//mem+=N*N*M*Nzp*sizeof(float);
					cudaMemPrefetchAsync(fn0s, N * N * M * Nzp * sizeof(float), igpu, s2);											//mem+=N*N*M*Nzp*sizeof(float);
					cudaMemPrefetchAsync(&ft0s[N * N * M], N * N * M * (Nzp - (iz + 1 == Nz / Nzp - 1)) * sizeof(float), igpu, s2); //mem+=N*N*M*(Nzp-(iz+1==Nz/Nzp-1))*sizeof(float);
					cudaMemPrefetchAsync(ftn0s, N * N * M * Nzp * sizeof(float), igpu, s2);											//mem+=N*N*M*Nzp*sizeof(float);
					cudaMemPrefetchAsync(h10s, N * Ntheta * Nzp * sizeof(float), igpu, s2);											//mem+=N*Ntheta*Nzp*sizeof(float);
					cudaMemPrefetchAsync(h20s, (N + 1) * (N + 1) * (M + 1) * (Nzp + 1) * sizeof(float4), igpu, s2);					//mem+=(N+1)*(N+1)*(M+1)*(Nzp+1)*sizeof(float4);
					cudaMemPrefetchAsync(g0s, N * Ntheta * Nzp * sizeof(float), igpu, s2);											//mem+=N*Ntheta*Nzp*sizeof(float);

					cudaEventRecord(e2, s2);
				}

				cudaMemPrefetchAsync(f0, N * N * M * Nzp * sizeof(float), cudaCpuDeviceId, s1);																												   //mem+=N*N*M*Nzp*sizeof(float);
				cudaMemPrefetchAsync(fn0, N * N * M * Nzp * sizeof(float), cudaCpuDeviceId, s1);																											   //mem+=N*N*M*Nzp*sizeof(float);
				cudaMemPrefetchAsync(&ft0[-(iz != 0) * N * N * M], N * N * M * (Nzp - (iz == 0) - (iz == Nz / Nzp - 1) + 2 * (iz == (igpu + 1) * Nz / Nzp / ngpus - 1)) * sizeof(float), cudaCpuDeviceId, s1); //mem+= N*N*M*(Nzp-(iz==0)-(iz==Nz/Nzp-1)+2*(iz==(igpu+1)*Nz/Nzp/ngpus-1))*sizeof(float);

				cudaMemPrefetchAsync(ftn0, N * N * M * Nzp * sizeof(float), cudaCpuDeviceId, s1);						  //mem+=N*N*M*Nzp*sizeof(float);
				cudaMemPrefetchAsync(h10, N * Ntheta * Nzp * sizeof(float), cudaCpuDeviceId, s1);						  //mem+=N*Ntheta*Nzp*sizeof(float);
				cudaMemPrefetchAsync(h20, (N + 1) * (N + 1) * (M + 1) * (Nzp + 1) * sizeof(float4), cudaCpuDeviceId, s1); //mem+=(N+1)*(N+1)*(M+1)*(Nzp+1)*sizeof(float4);
				cudaMemPrefetchAsync(g0, N * Ntheta * Nzp * sizeof(float), cudaCpuDeviceId, s1);						  //mem+=N*Ntheta*Nzp*sizeof(float);

				f0 = f0s;
				fn0 = fn0s;
				ft0 = ft0s;
				ftn0 = ftn0s;
				h10 = h10s;
				h20 = h20s;
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
				tmp = f;
				f = fn;
				fn = tmp;

				float norm[2] = {};
				for (int k = 0; k < N * Ntheta * Nz; k++)
					norm[0] += (h1[k] - g[k]) * (h1[k] - g[k]);
				for (int k = 0; k < (N + 1) * (N + 1) * (M + 1) * (Nzp + 1) * Nz / Nzp; k++)
					norm[1] += sqrt(h2[k].x * h2[k].x + h2[k].y * h2[k].y + h2[k].z * h2[k].z + h2[k].w * h2[k].w);
				fprintf(stderr, "iterations (%d/%d) f:%f, r:%f, total:%f\n", iter, niter, norm[0], lambda0 * norm[1], norm[0] + lambda0 * norm[1]);
				fflush(stdout);
			}
		}
		cudaDeviceSynchronize();
#pragma omp barrier
	}
	float end = omp_get_wtime();
	printf("Elapsed time: %fs.\n", end - start);
	cudaMemPrefetchAsync(ft, N * N * M * Nz * sizeof(float), cudaCpuDeviceId, 0);
	cudaMemcpy(fres, ft, N * N * M * Nz * sizeof(float), cudaMemcpyDefault);
}

void rectv::run_wrap(float *fres, int N0, float *g, int N1, float *theta, int N2, float *phi, int N3, size_t niter)
{
	run(fres, g, theta, phi, niter);
}