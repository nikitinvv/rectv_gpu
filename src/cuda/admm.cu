
#include "rectv.cuh"



// void rectv::solver_admm(float *f0, float *fn0, float *ft0, float *ftn0, float *h10, float4 *h20, float *g0,  int iz, int igpu, cudaStream_t s)
// {
// 	//forward step
// 	gradient(h20, ft0, iz, igpu, s); //iz for border control
// 	radonapr(h10, ft0, igpu, s);
// 	//proximal
// 	prox(h10, h20, g0, igpu, s);
// 	//backward step
// 	divergent(fn0, f0, h20, igpu, s);
// 	radonapradj(fn0, h10, igpu, s);
// 	//update ft
// 	updateft(ftn0, fn0, f0, igpu, s);
// }