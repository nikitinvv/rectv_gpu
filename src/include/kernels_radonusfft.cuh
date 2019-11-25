#define PI 3.1415926535

// Divide by phi
void __global__ divphi(float2 *g, float2 *f, float mu, int m, int n, int nz, dir direction)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= n || ty >= n || tz >= nz)
		return;
	float phi = __expf(-mu * (tx - n / 2) * (tx - n / 2) - mu * (ty - n / 2) * (ty - n / 2));
	int f_ind = (+tx + ty * n + tz * n * n);
	int g_ind = (+(tx + n / 2 + m) + (ty + n / 2 + m) * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m));
	if (direction == TOMO_FWD)
	{
		g[g_ind].x = f[f_ind].x / phi/ (4 * n * n);
		g[g_ind].y = f[f_ind].y / phi / (4 * n * n);
	}
	else
	{
		f[f_ind].x = g[g_ind].x / phi / (4 * n * n);
		f[f_ind].y = g[g_ind].y / phi / (4 * n * n);
	}
}

void __global__ fftshift(float2 *f, int n, int ntheta, int nz, bool flg)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= n || ty >= ntheta || tz >= nz)
		return;
	int g = (1 - 2 * ((tx + 1) % 2));
	if (flg)
		g *= (1 - 2 * ((ty + 1) % 2));
	f[tx + ty * n + tz * n * ntheta].x *= g;
	f[tx + ty * n + tz * n * ntheta].y *= g;
}

void __global__ wrap(float2 *f, int n, int nz, int m, dir direction)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= 2 * n + 2 * m || ty >= 2 * n + 2 * m || tz >= nz)
		return;
	if (tx < m || tx >= 2 * n + m || ty < m || ty >= 2 * n + m)
	{
		int tx0 = (tx - m + 2 * n) % (2 * n);
		int ty0 = (ty - m + 2 * n) % (2 * n);
		int id1 = tx + ty * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
		int id2 = tx0 + m + (ty0 + m) * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
		if (direction == TOMO_FWD)
		{
			f[id1].x = f[id2].x;
			f[id1].y = f[id2].y;
		}
		else
		{
			atomicAdd(&f[id2].x, f[id1].x);
			atomicAdd(&f[id2].y, f[id1].y);
		}
	}
}

void __global__ takexy(float *x, float *y, float *theta, int n, int ntheta)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tx >= n || ty >= ntheta)
		return;
	x[tx + ty * n] = (tx - n / 2) / (float)n * __sinf(theta[ty]);
	y[tx + ty * n] = (tx - n / 2) / (float)n * __cosf(theta[ty]);
	if (x[tx + ty * n] >= 0.5f)
		x[tx + ty * n] = 0.5f - 1e-5;
	if (y[tx + ty * n] >= 0.5f)
		y[tx + ty * n] = 0.5f - 1e-5;
}

void __global__ gather(float2 *g, float2 *f, float *x, float *y, int m,
					   float mu, int n, int ntheta, int nz, dir direction)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx >= n || ty >= ntheta || tz >= nz)
		return;

	float2 g0;
	float x0 = x[tx + ty * n];
	float y0 = y[tx + ty * n];
	int g_ind = tx + ty * n + tz * n * ntheta;
	if (direction == TOMO_FWD)
	{
		g0.x = 0.0f;
		g0.y = 0.0f;
	}
	else
	{
		g0.x = g[g_ind].x / n;
		g0.y = g[g_ind].y / n;
		if (tx == 0)
		{
			g0.x = 0;
			g0.y = 0;
		}
	}
	for (int i1 = 0; i1 < 2 * m + 1; i1++)
	{
		int ell1 = floorf(2 * n * y0) - m + i1;
		for (int i0 = 0; i0 < 2 * m + 1; i0++)
		{
			int ell0 = floorf(2 * n * x0) - m + i0;
			float w0 = ell0 / (float)(2 * n) - x0;
			float w1 = ell1 / (float)(2 * n) - y0;
			float w = PI / mu * __expf(-PI * PI / mu * (w0 * w0 + w1 * w1));
			int f_ind = n + m + ell0 + (2 * n + 2 * m) * (n + m + ell1) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
			if (direction == TOMO_FWD)
			{
				g0.x += w * f[f_ind].x;
				g0.y += w * f[f_ind].y;
			}
			else
			{
				float *fx = &(f[f_ind].x);
				float *fy = &(f[f_ind].y);
				atomicAdd(fx, w * g0.x);
				atomicAdd(fy, w * g0.y);
			}
		}
	}
	if (direction == TOMO_FWD)
	{
		g[g_ind].x = g0.x/ n;
		g[g_ind].y = g0.y/ n;
		if (tx == 0)
		{
			g[g_ind].x = 0;
			g[g_ind].y = 0;
		}
	}
}

void __global__ circ(float2 *f, float r, int n, int nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= n || ty >= n || tz >= nz)
		return;
	int id0 = tx + ty * n + tz * n * n;
	float x = (tx - n / 2) / float(n);
	float y = (ty - n / 2) / float(n);
	int lam = (4 * x * x + 4 * y * y) < 1 - r;
	f[id0].x *= lam;
	f[id0].y *= lam;
}

void __global__ mulr(float2 *f, float r, int n, int ntheta, int nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= n || ty >= ntheta || tz >= nz)
		return;
	int id0 = tx + ty * n + tz * ntheta * n;
	f[id0].x *= r;
	f[id0].y *= r;
}

void __global__ applyfilter(float2 *f, int n, int ntheta, int nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= n || ty >= ntheta || tz >= nz)
		return;
	int id0 = tx + ty * n + tz * ntheta * n;
	float rho = (tx - n / 2) / (float)n;
	float w = 0;
	if (rho == 0)
		w = 0;
	else
		w = abs(rho) * n * 4 * sin(rho) / rho; //(1-fabs(rho)/coef)*(1-fabs(rho)/coef)*(1-fabs(rho)/coef);
	f[id0].x *= w;
	f[id0].y *= w;
}

void __global__ takeshift(float2 *shift, float c, int n)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tx >= n)
		return;
	shift[tx].x = __cosf(2 * PI * c * (tx - n / 2.0) / n);
	shift[tx].y = __sinf(2 * PI * c * (tx - n / 2.0) / n);
}

void __global__ shift(float2 *f, float2 *shift, int n, int ntheta, int nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= n || ty >= ntheta || tz >= nz)
		return;
	float cr = shift[tx].x;
	float ci = shift[tx].y;
	int f_ind = tx + ty * n + tz * n * ntheta;
	float2 f0;
	f0.x = f[f_ind].x;
	f0.y = f[f_ind].y;
	f[f_ind].x = f0.x * cr - f0.y * ci;
	f[f_ind].y = f0.x * ci + f0.y * cr;
}