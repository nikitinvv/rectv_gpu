#include"rectv.cuh"
#include<math.h>
#include<stdio.h>
#include<string.h>

int main(int argc,char *argv[])
{
	const char* parfile = argv[1];
	const char* datafile = argv[2];
	const char* resfile = argv[3];

//pars
	size_t N,Nrot,Ntheta,M,Nz,Nzp,ngpus,niters;
	float lambda0,lambda1;

	FILE* fid = fopen(parfile,"r");
	fscanf(fid,"%ld %ld %ld %ld %ld %ld %ld %ld\n",&N,&Nrot,&Ntheta,&M,&Nz,&Nzp,&ngpus,&niters);
	fscanf(fid,"%e %e\n",&lambda0,&lambda1);
	fclose(fid);

	fprintf(stderr,"%ld %ld %ld %ld %ld %ld %ld %ld\n",N,Nrot,Ntheta,M,Nz,Nzp,ngpus,niters);
	fprintf(stderr,"%e %e\n",lambda0,lambda1);

//data

	float* g = new float[N*Ntheta*Nz];		
	float* fres = new float[N*N*M*Nz];		
	fid = fopen(datafile,"rb");
	fread(g,4,N*Ntheta*Nz,fid);

//class to perform the regularization
	rectv* r0 = new rectv(N,Ntheta,M,Nrot,Nz,Nzp,ngpus,lambda0,lambda1);
//run iterations
	r0->itertvR(fres,g,niters);

	fid = fopen(resfile,"wb");
	fwrite(fres,4,N*N*M*Nz,fid);
	fclose(fid);

	delete[] g;
	delete[] fres;
	delete r0;
	
	return 0;
}

