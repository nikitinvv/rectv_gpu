#include <stdio.h>
#include <omp.h>
#include "rectv.cuh"
#include "kernels.cuh"

rectv::rectv(size_t N_, size_t Ntheta_, size_t M_, size_t Nrot_, size_t Nz_, size_t Nzp_, size_t ngpus_, float lambda0_, float lambda1_)
{
	N = N_;
	Ntheta = Ntheta_;
	M = M_;
	Nrot = Nrot_;
	Nz = Nz_;
	Nzp = Nzp_;
	lambda0 = lambda0_;
	lambda1 = lambda1_;
	ngpus = min(ngpus_, (size_t)(Nz / Nzp));
	tau = 1 / sqrt(1 + 1 + lambda1 * lambda1 + (Nz != 0)); //sqrt norm of K1^*K1+K2^*K2
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
	gtmps = new float *[ngpus];
	phi = new float2 *[ngpus];
	theta = new float *[ngpus];

	dim3 BS2d(32, 32);
	dim3 GS2d0(ceil(Ntheta / (float)BS2d.x), ceil(M / (float)BS2d.y));

	for (int igpu = 0; igpu < ngpus; igpu++)
	{
		cudaSetDevice(igpu);
		rad[igpu] = new radonusfft(N, Ntheta / Nrot, Nzp);
		cudaMalloc((void **)&ftmp[igpu], 2 * (N + 2) * (N + 2) * (M + 2) * (Nzp + 2) * sizeof(float));
		cudaMalloc((void **)&gtmp[igpu], 2 * N * Ntheta * Nzp * sizeof(float));
		cudaMalloc((void **)&ftmps[igpu], 2 * N * N * Nzp * sizeof(float));
		cudaMalloc((void **)&gtmps[igpu], 2 * N * Ntheta / Nrot * Nzp * sizeof(float));
		cudaMalloc((void **)&phi[igpu], Ntheta * M * sizeof(float2));
		cudaMalloc((void **)&theta[igpu], Ntheta / Nrot * sizeof(float));
		//angles [0,pi]
		taketheta<<<ceil(Ntheta / Nrot / 1024.0), 1024>>>(theta[igpu], Ntheta, Nrot);
		takephi<<<GS2d0, BS2d>>>(phi[igpu], Ntheta, M);
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
		cudaFree(gtmps[igpu]);
		cudaFree(phi[igpu]);
		cudaFree(theta[igpu]);
		cudaDeviceReset();
	}
}

void rectv::radonapr(float *g, float *f, int igpu, cudaStream_t s)
{
	//tmp arrays on gpus
	float2 *ftmp0 = (float2 *)ftmp[igpu];
	float2 *ftmps0 = (float2 *)ftmps[igpu];
	float2 *gtmp0 = (float2 *)gtmp[igpu];
	float2 *gtmps0 = (float2 *)gtmps[igpu];
	float2 *phi0 = (float2 *)phi[igpu];
	float *theta0 = (float *)theta[igpu];

	cudaMemsetAsync(ftmp0, 0, 2 * N * N * M * Nzp * sizeof(float), s);
	cudaMemsetAsync(ftmps0, 0, 2 * N * N * Nzp * sizeof(float), s);
	cudaMemsetAsync(gtmp0, 0, 2 * N * Ntheta * Nzp * sizeof(float), s);
	cudaMemsetAsync(gtmps0, 0, 2 * N * Ntheta / Nrot * Nzp * sizeof(float), s);

	dim3 BS3d(32, 32, 1);
	dim3 GS3d0(ceil(N / (float)BS3d.x), ceil(N / (float)BS3d.y), ceil(Nzp * M / (float)BS3d.z));
	dim3 GS3d1(ceil(N / (float)BS3d.x), ceil(N / (float)BS3d.y), ceil(Nzp / (float)BS3d.z));
	dim3 GS3d2(ceil(N / (float)BS3d.x), ceil(Ntheta / (float)BS3d.y), ceil(Nzp / (float)BS3d.z));
	//switch to complex numbers
	makecomplexf<<<GS3d0, BS3d, 0, s>>>(ftmp0, f, N, M, Nzp);
	for (int i = 0; i < M; i++)
	{
		//decompositon coefficients
		decphi<<<GS3d1, BS3d, 0, s>>>(ftmps0, ftmp0, &phi0[i * Ntheta], N, Ntheta, M, Nzp);
		//Radon tranform for [0,pi) interval
		rad[igpu]->fwdR(gtmps0, ftmps0, theta0, s);

		//spread Radon data over all angles
		for (int k = 0; k < Nrot; k++)
			copys<<<GS3d2, BS3d, 0, s>>>(&gtmp0[k * N * Ntheta / Nrot], gtmps0, k % 2, N, Ntheta, Ntheta / Nrot, Nzp);
		//constant for normalization
		mulc<<<GS3d2, BS3d, 0, s>>>(gtmp0, 1.0f / sqrt(M) * Ntheta / sqrt(Nrot), N, Ntheta, Nzp);
		//multiplication by basis functions
		mulphi<<<GS3d2, BS3d, 0, s>>>(gtmp0, &phi0[i * Ntheta], 1, N, Ntheta, Nzp); //-1 conj
		//sum up
		addg<<<GS3d2, BS3d, 0, s>>>(g, gtmp0, tau, N, Ntheta, Nzp);
	}
}

void rectv::radonapradj(float *f, float *g, int igpu, cudaStream_t s)
{
	//tmp arrays on gpus
	float2 *ftmp0 = (float2 *)ftmp[igpu];
	float2 *ftmps0 = (float2 *)ftmps[igpu];
	float2 *gtmp0 = (float2 *)gtmp[igpu];
	float2 *gtmps0 = (float2 *)gtmps[igpu];
	float2 *phi0 = (float2 *)phi[igpu];
	float *theta0 = (float *)theta[igpu];

	cudaMemsetAsync(ftmp0, 0, 2 * N * N * M * Nzp * sizeof(float), s);
	cudaMemsetAsync(ftmps0, 0, 2 * N * N * Nzp * sizeof(float), s);
	cudaMemsetAsync(gtmp0, 0, 2 * N * Ntheta * Nzp * sizeof(float), s);
	cudaMemsetAsync(gtmps0, 0, 2 * N * Ntheta / Nrot * Nzp * sizeof(float), s);

	dim3 BS3d(32, 32, 1);
	dim3 GS3d0(ceil(N / (float)BS3d.x), ceil(N / (float)BS3d.y), ceil(Nzp * M / (float)BS3d.z));
	dim3 GS3d1(ceil(N / (float)BS3d.x), ceil(N / (float)BS3d.y), ceil(Nzp / (float)BS3d.z));
	dim3 GS3d2(ceil(N / (float)BS3d.x), ceil(Ntheta / (float)BS3d.y), ceil(Nzp / (float)BS3d.z));
	for (int i = 0; i < M; i++)
	{
		//switch to complex numbers
		makecomplexR<<<GS3d2, BS3d, 0, s>>>(gtmp0, g, N, Ntheta, Nzp);
		//multiplication by conjugate basis functions
		mulphi<<<GS3d2, BS3d, 0, s>>>(gtmp0, &phi0[i * Ntheta], -1, N, Ntheta, Nzp); //-1 conj
		//constant for normalization
		mulc<<<GS3d2, BS3d, 0, s>>>(gtmp0, 1.0f / sqrt(M) * Ntheta / sqrt(Nrot), N, Ntheta, Nzp);
		//gather Radon data over all angles
		cudaMemsetAsync(gtmps0, 0, 2 * N * Ntheta / Nrot * Nzp * sizeof(float), s);
		for (int k = 0; k < Nrot; k++)
			adds<<<GS3d2, BS3d, 0, s>>>(gtmps0, &gtmp0[k * N * Ntheta / Nrot], k % 2, N, Ntheta, Ntheta / Nrot, Nzp);
		//adjoint Radon tranform for [0,pi) interval
		rad[igpu]->adjR(ftmps0, gtmps0, theta0, 0, s);

		//recovering by coefficients
		recphi<<<GS3d1, BS3d, 0, s>>>(ftmp0, ftmps0, &phi0[i * Ntheta], N, Ntheta, M, Nzp);
	}
	addf<<<GS3d0, BS3d, 0, s>>>(f, ftmp0, tau, N, M, Nzp);
}

void rectv::gradient(float4 *h2, float *ft, int iz, int igpu, cudaStream_t s)
{
	dim3 BS3d(32, 32, 1);
	dim3 GS3d0(ceil((N + 2) / (float)BS3d.x), ceil((N + 2) / (float)BS3d.y), ceil((M + 2) * (Nzp + 2) / (float)BS3d.z));
	float *ftmp0 = ftmp[igpu];
	//repeat border values
	extendf<<<GS3d0, BS3d, 0, s>>>(ftmp0, ft, iz != 0, iz != Nz / Nzp - 1, N + 2, M + 2, Nzp + 2);
	grad<<<GS3d0, BS3d, 0, s>>>(h2, ftmp0, tau, lambda1, N + 1, M + 1, Nzp + 1);
}

void rectv::divergent(float *fn, float *f, float4 *h2, int igpu, cudaStream_t s)
{
	dim3 BS3d(32, 32, 1);
	dim3 GS3d0(ceil(N / (float)BS3d.x), ceil(N / (float)BS3d.y), ceil(M * Nzp / (float)BS3d.z));
	div<<<GS3d0, BS3d, 0, s>>>(fn, f, h2, tau, lambda1, N, M, Nzp);
}

void rectv::prox(float *h1, float4 *h2, float *g, int igpu, cudaStream_t s)
{
	dim3 BS3d(32, 32, 1);
	dim3 GS3d0(ceil((N + 1) / (float)BS3d.x), ceil((N + 1) / (float)BS3d.y), ceil((M + 1) * (Nzp + 1) / (float)BS3d.z));
	dim3 GS3d1(ceil(N / (float)BS3d.x), ceil(Ntheta / (float)BS3d.y), ceil(Nzp / (float)BS3d.z));
	prox1<<<GS3d1, BS3d, 0, s>>>(h1, g, tau, N, Ntheta, Nzp);
	prox2<<<GS3d0, BS3d, 0, s>>>(h2, lambda0, N + 1, M + 1, Nzp + 1);
}

void rectv::updateft(float *ftn, float *fn, float *f, int igpu, cudaStream_t s)
{
	dim3 BS3d(32, 32, 1);
	dim3 GS3d0(ceil(N / (float)BS3d.x), ceil(N / (float)BS3d.y), ceil(M * Nzp / (float)BS3d.z));
	updateft_ker<<<GS3d0, BS3d, 0, s>>>(ftn, fn, f, N, M, Nzp);
}

void rectv::radonfbp(float *f, float *g, int igpu, cudaStream_t s)
{
	//tmp arrays on gpus
	float2 *ftmp0 = (float2 *)ftmp[igpu];
	float2 *gtmp0 = (float2 *)gtmp[igpu];
	float2 *gtmps0 = (float2 *)gtmps[igpu];
	float *theta0 = (float *)theta[igpu];

	cudaMemsetAsync(gtmp0, 0, 2 * N * Ntheta * Nzp * sizeof(float), s);
	dim3 BS3d(32, 32, 1);
	dim3 GS3d0(ceil(N / (float)BS3d.x), ceil(N / (float)BS3d.y), ceil(Nzp * M / (float)BS3d.z));
	dim3 GS3d1(ceil(N / (float)BS3d.x), ceil(N / (float)BS3d.y), ceil(Nzp / (float)BS3d.z));
	dim3 GS3d2(ceil(N / (float)BS3d.x), ceil(Ntheta / (float)BS3d.y), ceil(Nzp / (float)BS3d.z));

	//switch to complex numbers
	makecomplexR<<<GS3d2, BS3d, 0, s>>>(gtmp0, g, N, Ntheta, Nzp);
	for (int k = 0; k < Nrot; k++)
	{
		cudaMemsetAsync(gtmps0, 0, 2 * N * Ntheta / Nrot * Nzp * sizeof(float), s);
		adds<<<GS3d2, BS3d, 0, s>>>(gtmps0, &gtmp0[k * N * Ntheta / Nrot], k % 2, N, Ntheta, Ntheta / Nrot, Nzp);
		//adjoint Radon tranform for [0,pi) interval
		rad[igpu]->adjR(ftmp0, gtmps0, theta0, 1, s); //filter=1
		makerealstepf<<<GS3d1, BS3d, 0, s>>>(&f[N * N * k], ftmp0, N, Nrot, Nzp);
	}
	//constant for fidelity
	mulr<<<GS3d0, BS3d, 0, s>>>(f, 1 / sqrt(2 * M / (float)Nrot), N, M, Nzp);
}

void rectv::itertvR(float *fres, float *g_, size_t niter)
{
	cudaMemcpy(g, g_, N * Ntheta * Nz * sizeof(float), cudaMemcpyHostToHost);
	//take fbp as a first guess
#pragma omp parallel for
	for (int iz = 0; iz < Nz / Nzp; iz++)
	{
		int igpu = omp_get_thread_num();
		cudaSetDevice(igpu);
		float *f0 = &f[N * N * M * iz * Nzp];
		float *ft0 = &ft[N * N * M * iz * Nzp];
		float *g0 = &g[N * Ntheta * iz * Nzp];
		radonfbp(ft0, g0, igpu, 0);
		//spread results for all M
		for (int izp = 0; izp < Nzp; izp++)
			for (int i = 0; i < M; i++)
				cudaMemcpy(&f0[N * N * i + izp * N * N * M], &ft0[N * N * (i / (M / Nrot)) + N * N * Nrot * izp], N * N * sizeof(float), cudaMemcpyHostToHost);
	}

	cudaMemcpy(ft, f, N * N * M * Nz * sizeof(float), cudaMemcpyHostToHost);
	cudaMemcpy(fn, f, N * N * M * Nz * sizeof(float), cudaMemcpyHostToHost);
	cudaMemcpy(ftn, f, N * N * M * Nz * sizeof(float), cudaMemcpyHostToHost);
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
#pragma omp for
			for (int iz = 0; iz < Nz / Nzp; iz++)
			{
				cudaEventSynchronize(e1);
				cudaEventSynchronize(e2);

				//forward step
				gradient(h20, ft0, iz, igpu, s1); //iz for border control
				radonapr(h10, ft0, igpu, s1);
				//proximal
				prox(h10, h20, g0, igpu, s1);
				//backward step
				divergent(fn0, f0, h20, igpu, s1);
				radonapradj(fn0, h10, igpu, s1);
				//update ft
				updateft(ftn0, fn0, f0, igpu, s1);
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
				fprintf(stderr, "iterations (%d/%d) \r", iter, niter);
				fflush(stdout);
			}
		}
		cudaDeviceSynchronize();
#pragma omp barrier
	}
	float end = omp_get_wtime();
	printf("Elapsed time: %fs.\n", end - start);
	cudaMemPrefetchAsync(ft, N * N * M * Nz * sizeof(float), cudaCpuDeviceId, 0);
	float mcons = sqrt((float)M / Nrot / 4);
	for (int i = 0; i < N * N * M * Nz; i++)
		ft[i] *= mcons;
	cudaMemcpy(fres, ft, N * N * M * Nz * sizeof(float), cudaMemcpyDefault);
}

void rectv::itertvR_wrap(float *fres, int N0, float *g_, int N1, size_t niter)
{
	itertvR(fres, g_, niter);
}
