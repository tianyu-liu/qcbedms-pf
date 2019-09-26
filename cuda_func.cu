/*
 * CUDA functions of QCBEDMS-PF
 */
#include <cufft.h>
//#include <cuda_profiler_api.h>
#include "cpp_comm.hpp"

using namespace std;
typedef cuDoubleComplex CD;
typedef cuFloatComplex CF;

__constant__ int mx, my, mthf, meshx, meshy, mt, nslice, mbout, ntilt;

template<typename T>
__host__ __device__ static __inline__ T CmplxMul(T a, T b)
{
	return { a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x };
}

bool debug = true;
void cu_dbg(string msg, cudaError_t err)
{
	if(debug && err < 0) printf("%s: %s\n", msg.c_str(), cudaGetErrorString(err));
}

struct cufft_vars {
	void *pmap, *pg, *pr;
	void *tiltarray;
	void *beam;
	int *ib;
	cufftHandle plan;
	dim3 g_propg, b_propg, g_pmap, b_pmap, g_beam, b_beam;
	size_t g_pad, b_pad;
	complex<float> *host_pgf;
	float *host_tiltarrayf, *host_beamf;
} *cuv;

extern "C"
{
void cufft_init()
{
	cuv = new cufft_vars[MSP.nhyb];
}

void cufft_term()
{
	delete[] cuv;
}

void CUDAGetDevices(int in, int *count)
{
	cudaGetDeviceCount(count);
	if (in == -2)
		return;

	printf("Found %i CUDA device(s):\n", *count);
	cudaDeviceProp prop;
	for (int i = 0 ; i < *count ; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("\t%i. %s, compute capability %d.%d\n",
				i, prop.name, prop.major, prop.minor);
	}
	exit(0);
}

void cufft_child_init(const int f)
{
/* Choose CUDA device */
	cudaDeviceProp prop;
	cudaSetDevice(MSP.dev[f]);
	cudaGetDeviceProperties(&prop, MSP.dev[f]);
	cu_dbg("set dev", cudaSetDeviceFlags(cudaDeviceScheduleYield));
	fortran_write(13, "  Device: %d. %s.\n", MSP.dev[f], prop.name);

/* Measure cuFFT working memory and optional auto-determine beam pixels split */
	fortran_write(13, "  cuFFT precision: %s.\n", MSP.sgl ? "single" : "double");
	float MB = 1.0 / 1024 / 1024;
	size_t free_gm, total_gm;
	cudaError_t mem_err = cudaMemGetInfo(&free_gm, &total_gm);
	if (MSP.splt[f] == 0) {
		MSP.splt[f] = 1, MSP.ntilt[f] = MSP.as_nt[f];
		while (msp_mem(f) > free_gm && MSP.splt[f] < MSP.as_nt[f])
			MSP.ntilt[f] = (int) ceil((float) MSP.as_nt[f] / ++MSP.splt[f]);
		fortran_write(13, "(AUTO) ");
	}
	else
		MSP.ntilt[f] = (int) ceil((float) MSP.as_nt[f] / MSP.splt[f]);

	fortran_write(13, "  Paralleling %d/%d tilts; will loop %d time(s).\n",
			MSP.ntilt[f], MSP.as_nt[f], MSP.splt[f]);
	fortran_write(13, "  Available GPU memory: %g/%g MB, cuFFT will use at least %g MB.\n",
			free_gm * MB, total_gm * MB, msp_mem(f) * MB);

	if (msp_mem(f) > free_gm) {
		if (F.ncalcs < 1) {
			fortran_write(13, "\nNOT enough GPU memory! "
					"Please reduce paralleled tilts manually.\n");
			cout << "NOT enough GPU memory! Please reduce paralleled tilts manually." << endl;
			exit(-1);
		} else if (mem_err != cudaSuccess) {
			fortran_write(13, "\nWARNING: \"CUDA get memory usage\" failed "
					"during refinement! Ignored and continue.\n");
			fortran_write(13, "%s\n", cudaGetErrorString(mem_err));
		}
	}
	if (msp_mem(f) > free_gm * 0.5)
		fortran_write(13, "  GPU memory may be intensive; reduce paralleled "
				"tilts manually if any error occurs.\n");

/* Set global constants (even before choosing devices) */
	int mxt = F.meshx / 2, myt = F.meshy / 2, mthft = F.mt / 2;
//	printf("mx=%i\n",mxt);
	cudaMemcpyToSymbolAsync(mx, &mxt, sizeof(int));
	cudaMemcpyToSymbolAsync(my, &myt, sizeof(int));
	cudaMemcpyToSymbolAsync(mthf, &mthft, sizeof(int));
	cudaMemcpyToSymbol(meshx, &F.meshx, sizeof(int));
	cudaMemcpyToSymbolAsync(meshy, &F.meshy, sizeof(int));
	cudaMemcpyToSymbolAsync(mt, &F.mt, sizeof(int));
	cudaMemcpyToSymbolAsync(nslice, &F.nslice, sizeof(int));
	cudaMemcpyToSymbolAsync(mbout, &F.mbout, sizeof(int));
	cudaMemcpyToSymbolAsync(ntilt, &MSP.ntilt[f], sizeof(int));

/* Allocate intermediate single-precision arrays on host */
	if(MSP.sgl) {
		cuv[f].host_pgf = new complex<float>[F.mt * F.nslice];
		cuv[f].host_tiltarrayf = new float[2 * F.tottilts * F.npr];
		cuv[f].host_beamf = new float[F.mbout * MSP.ntilt[f] * size_R];
	}
/* Allocate GPU arrays */
	cudaMalloc((void **) &cuv[f].pg, F.mt * F.nslice * size_C);
	cudaMalloc((void **) &cuv[f].pr, F.mt * MSP.ntilt[f] * F.npr * size_C);
	cudaMalloc((void **) &cuv[f].pmap, F.mt * MSP.ntilt[f] * size_C);
	cudaMalloc((void **) &cuv[f].tiltarray, 2 * MSP.ntilt[f] * size_R);
	cudaMalloc((void **) &cuv[f].beam, MSP.ntilt[f] * F.mbout * size_R);
	cudaMalloc((void **) &cuv[f].ib, F.mbout * sizeof(int));
	cudaMemcpyAsync(cuv[f].ib, F.ib, F.mbout * sizeof(int), cudaMemcpyHostToDevice, 0);
	cudaMemsetAsync(cuv[f].beam, 0, MSP.ntilt[f] * F.mbout * size_R);

/* Setup batched 2D cuFFT plan */
	int n[2] = { (int) F.meshy, (int) F.meshx };
	cufftPlanMany(&cuv[f].plan, 2, n, NULL, 1, F.mt, NULL, 1, F.mt,
			MSP.sgl ? CUFFT_C2C : CUFFT_Z2Z, MSP.ntilt[f]);

	int maxThreads, warp_size, SMs, attr[100];
	for (size_t i = 1 ; i < 92 ; i++)
		cudaDeviceGetAttribute(&attr[i], (cudaDeviceAttr) i, 0);
	maxThreads = attr[1];
	warp_size = attr[10];
	SMs = attr[16];

/* Kernel cu_propg GPU threads mapping
 1. 0 <= threadIdx.x + blockIdx.y * blockDim.x < meshx,
	0 <= threadIdx.y + blockIdx.z * blockDim.y < meshy;
 2. 0 <= blockIdx.x < MSP.ntilt[f], as gridDim.x can hold the most blocks;
 3. Repeat kernel for F.nslice.
 */
	cuv[f].b_propg = cuv[f].g_propg = 1;
	cuv[f].b_propg.x = F.meshx, cuv[f].b_propg.y = F.meshy;
	int i = 1;
	while ((cuv[f].b_propg.x * cuv[f].b_propg.y % warp_size != 0 &&
			cuv[f].b_propg.x * cuv[f].b_propg.y > warp_size) ||
			cuv[f].b_propg.x * cuv[f].b_propg.y > maxThreads) {
		if (i == 0)
			i++, cuv[f].b_propg.x /= 2, cuv[f].g_propg.y *= 2;
		else
			i--, cuv[f].b_propg.y /= 2, cuv[f].g_propg.z *= 2;
	}
	cuv[f].g_propg.x = MSP.ntilt[f];

/* Kernel pad0oshift GPU threads mapping
 1. 0 <= threadIdx.x + blockIdx.x * blockDim.x < MSP.ntilt[f];
 2. Repect kernel for F.nslice.
*/
	cuv[f].b_pad = ceil(1. * MSP.ntilt[f] / SMs);
	if (cuv[f].b_pad > maxThreads)
		cuv[f].b_pad = maxThreads;
	cuv[f].g_pad = ceil(1. * MSP.ntilt[f] / cuv[f].b_pad);

/* Kernels pmapXpg & pmapXpg GPU threads mapping
 1. 'cuv[f].b_pmap.x' holds 'F.mt' as threads per block,
 'cuv[f].g_pmap.x' holds 'MSP.ntilt[f]' as number of blocks;
 2. If 'cuv[f].b_pmap.x' > max allowed threads number 'maxThreads',
 keep halving it and the excess goes to grid 'cuv[f].g_pmap.y';
 3. Finally, try to make total-thread-number divisible by warp 'warp_size'.

 So threads iterate upon 'cuv[f].b_pmap.x' only, while blocks iterate through
 'cuv[f].g_pmap.x' * 'cuv[f].g_pmap.y'.

 Criteria:
 1. 0 <= threadIdx.x * gridDim.y + blockIdx.y < cuv[f].b_pmap.x * cuv[f].g_pmap.y = F.mt,
 they control mesh iteration of every pixel;
 2. 0 <= blockIdx.x < cuv[f].g_pmap.x = MSP.ntilt[f], for pixel iterations.
 */

	cuv[f].b_pmap = cuv[f].g_pmap = 1;
	cuv[f].b_pmap.x = F.mt, cuv[f].g_pmap.x = MSP.ntilt[f];
	while ((cuv[f].b_pmap.x % warp_size != 0 && cuv[f].b_pmap.x > warp_size) ||
			cuv[f].b_pmap.x > maxThreads)
		cuv[f].b_pmap.x /= 2, cuv[f].g_pmap.y *= 2;

/* Kernel calc_beam GPU threads mapping
 Total paralleled jobs = (num of beams 'F.mbout') * (paralleled pixels 'MSP.ntilt[f]').

 1. Primarily, divide 'MSP.ntilt[f]' iterations into blocks 'cuv[f].g_beam.x' same to number
 of multiprocessors, and put the threads in 'cuv[f].b_beam.x', but
 'cuv[f].b_beam.x' * 'cuv[f].g_beam.x' >= 'MSP.ntilt[f]';
 2. Map beam iterations to 'cuv[f].g_beam.y';
 3. If 'cuv[f].b_beam.x' > 'maxThreads', then maximise 'cuv[f].b_beam.x' to 'maxThreads',
 and map the excess jobs back to grid 'cuv[f].g_beam.x'

 So threads iterate upon 'cuv[f].b_beam.x' only, while blocks iterate through
 'cuv[f].g_beam.x' * 'cuv[f].g_beam.y'.

 Criteria:
 1. cuv[f].b_beam.x * cuv[f].g_beam.x = MSP.ntilt[f], they control 'MSP.ntilt[f]' iteration of each beam
 (threadIdx.x * gridDim.y + blockIdx.y);
 2. cuv[f].g_beam.y = blockIdx.x = F.mbout, for beam iterations.
 */

	cuv[f].b_beam = cuv[f].g_beam = 1;
	cuv[f].b_beam.x = ceil(1. * MSP.ntilt[f] / SMs);
	if (cuv[f].b_beam.x > maxThreads)
		cuv[f].b_beam.x = maxThreads;
	cuv[f].g_beam.x = ceil(1. * MSP.ntilt[f] / cuv[f].b_beam.x);
	cuv[f].g_beam.y = F.mbout;

//	printf("pmap block = %i %i %i, grid = %i %i %i\n",
//			cuv[f].b_pmap.x, cuv[f].b_pmap.y, cuv[f].b_pmap.z,
//			cuv[f].g_pmap.x, cuv[f].g_pmap.y, cuv[f].g_pmap.z);
//	printf("beam block = %i %i %i, grid = %i %i %i\n",
//			cuv[f].b_beam.x, cuv[f].b_beam.y, cuv[f].b_beam.z,
//			cuv[f].g_beam.x, cuv[f].g_beam.y, cuv[f].g_beam.z);

//	cudaDeviceSynchronize();
//	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}

void cufft_child_term(const int f)
{
	if(MSP.sgl) {
		delete[] cuv[f].host_pgf;
		delete[] cuv[f].host_tiltarrayf;
		delete[] cuv[f].host_beamf;
	}
	cudaSetDevice(MSP.dev[f]);
	cudaFree(cuv[f].pg);
	cudaFree(cuv[f].pr);
	cudaFree(cuv[f].pmap);
	cudaFree(cuv[f].beam);
	cudaFree(cuv[f].ib);
	cudaFree(cuv[f].tiltarray);
	cufftDestroy(cuv[f].plan);
	cudaDeviceReset();
}
}

// CUDA kernels:
template<typename T2, typename T3>
__global__ void propg(const T3 xyz, const T3 c2r,
		const T2 *__restrict__ tiltarray, T2 *__restrict__ pr)
{
	int x = threadIdx.x + blockIdx.y * blockDim.x;
	int y = threadIdx.y + blockIdx.z * blockDim.y;
	int h = x - mx, k = y - my, t = blockIdx.x;
	T2 tilt = tiltarray[t];
	T2 ab = { tilt.x + h, tilt.y + k };
	T3 sc;
	sc.z = ab.x * h * c2r.x + ab.y * k * c2r.y + (ab.x * k + ab.y * h) * c2r.z;
	sincospi(2 * (sc.z * xyz.z + h * xyz.x + k * xyz.y), &sc.x, &sc.y);
	pr[x + y * meshx + t * mt] = { sc.y / mt, sc.x / mt };
//	if (t == 3 && x!=0&&y!=0)
//		printf("pr: %d %g %g\n", t, pr[x + y * meshx + t * mt].x,
//				pr[x + y * meshx + t * mt].y);
//	if (t == 1 && x!=0&&y!=0)
//		printf("meshx=%i meshy=%i mt=%i\n",meshx,meshy,mt);
//		printf("%i.%i %i %i %i %g %g\n", x,y, h,k,x+y*meshx,pr[x + y * meshx + t * mt].x,
//				pr[x + y * meshx + t * mt].y);
//	if (x == 0 && y == 0)
//		printf("%i %g %g\n", t,tilt.x,tilt.y);
}

template<typename T2>
__global__ void pad0oshift(T2* pr)
{
	int tmt = threadIdx.x + blockIdx.x * blockDim.x;
	if (tmt < ntilt) {
		tmt *= mt;
// PAD0
		for (int i = tmt ; i < meshx + tmt ; ++i)
			pr[i] = {0., 0.};
		for (int j = meshx + tmt ; j < mt + tmt ; j += meshx)
			pr[j] = {0., 0.};
// OSHIFT
		T2 tmp;
		for (int j = tmt ; j < mthf - meshx + tmt + 1 ; j += meshx)
			for (int i = j, i1 = i + mx + mthf ; i < mx + j ; ++i, ++i1)
				tmp = pr[i], pr[i] = pr[i1], pr[i1] = tmp;
		for (int j = mthf + tmt ; j < mt - meshx + tmt + 1 ; j += meshx)
			for (int i = j, i1 = i + mx - mthf ; i < mx + j ; ++i, ++i1)
				tmp = pr[i], pr[i] = pr[i1], pr[i1] = tmp;
//		if(tmt/mt==0)
//		for (int i = 0 ; i < mt ; ++i)
//			printf("%i-%i. %g %g\n",tmt/mt,i,pr[tmt+i].x*mt,pr[tmt+i].y*mt);
	}
}

template<typename T2>
__global__ void pmapCmplx1(T2 *__restrict__ pmap)
{
	int i = threadIdx.x + (blockIdx.y + blockIdx.x * gridDim.y) * blockDim.x;
	pmap[i].x = 1., pmap[i].y = 0.;
}

template<typename T2>
__global__ void pmapXpg(T2 *__restrict__ pmap, const T2 *__restrict__ pg)
{
	int i = threadIdx.x + blockIdx.y * blockDim.x;
	int j = blockIdx.x * mt + i;
	pmap[j] = CmplxMul(pmap[j], pg[i]);
//	if(j<1) {
//		printf("pmap %i %.15g %.15g\n", j, a.x / mt, a.y / mt);
//		printf("pg %i %.15g %.15g\n", j, b.x, b.y);
//		printf("pmap %i %.15g %.15g\n", j, pmap[j].x / mt, pmap[j].y / mt);
//	}
}

template<typename T2>
__global__ void pmapXpr(T2 *__restrict__ pmap, const T2 *__restrict__ pr)
{
	int i = threadIdx.x + (blockIdx.y + blockIdx.x * gridDim.y) * blockDim.x;
//	if(i<5) printf("pr %i %.15g %.15g\n",i,pr[i].x*mt,pr[i].y*mt);
	pmap[i] = CmplxMul(pmap[i], pr[i]);
}

template<typename T1, typename T2>
__global__ void calc_beam(T1 *__restrict__ beam, T2 *__restrict__ pmap, int *__restrict__ ib)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
	if (i < ntilt) {
		const T2 m = pmap[i * mt + ib[j] - 1];
		beam[i + j * ntilt] = (T1) m.x * m.x + m.y * m.y;
//		if(j<5) printf("pmap %i %.15g %.15g\n",j,pmap[j].x,pmap[j].y);
	}
}

template<typename T1, typename T2, typename T3>
void cufft_msp(const int f, const int ct1, const int ct2, complex<T1> *host_pg, T1 *host_tiltarray)
{
	time_point t_msp;
	cudaSetDevice(MSP.dev[f]);

	for (int isplt = 0 ; isplt < MSP.splt[f] ; ++isplt) {
		int tct1 = isplt * MSP.ntilt[f] + ct1;
		int tnt = min(MSP.ntilt[f], ct2 - tct1 + 1);
//		printf("%d %d %d %d %d %lu\n",ct1,ct2);

		cudaMemcpyAsync(cuv[f].pg, host_pg, F.mt * F.nslice * size_C, cudaMemcpyHostToDevice, 0);

//	cudaProfilerStart();
		if (Tr.msp) {
			cudaDeviceSynchronize();
			t_msp = time_now();
		}
		for (int i = 0 ; i < F.npr ; ++i) {
			cudaMemcpyAsync(cuv[f].tiltarray,
					host_tiltarray + 2 * (tct1 + i * F.tottilts),
					2 * tnt * size_R, cudaMemcpyHostToDevice, 0);
			T3 xyz = { (T1) F.xshift[i], (T1) F.yshift[i],
					(T1) (F.deltaz[i] * F.applieddilation[i]) };
			T3 c2r = { (T1) F.cell2rpr[3 * i], (T1) F.cell2rpr[3 * i + 1],
					(T1) F.cell2rpr[3 * i + 2] };
			propg<<<cuv[f].g_propg, cuv[f].b_propg>>>((T3) xyz, (T3) c2r, (T2*) cuv[f].tiltarray,
					((T2*) cuv[f].pr) + i * F.mt * MSP.ntilt[f] );

			pad0oshift<T2><<<cuv[f].g_pad, cuv[f].b_pad>>>(((T2*)cuv[f].pr) + i * F.mt * MSP.ntilt[f]);
		}
		if (Tr.msp) {
			cudaDeviceSynchronize();
			Tr.pr[f] += timer(&t_msp);
		}

		pmapCmplx1<T2><<<cuv[f].g_pmap,cuv[f].b_pmap>>>((T2*) cuv[f].pmap);

		if (Tr.msp) {
			cudaDeviceSynchronize();
			t_msp = time_now();
		}

		for (int i = 0, is ; i < F.sl_hi ; i++) {
			is = F.mseq[i] - 1;
			pmapXpg<T2> <<<cuv[f].g_pmap, cuv[f].b_pmap>>>(
					(T2*) cuv[f].pmap, ((T2*) cuv[f].pg) + is * F.mt);
			if (Tr.msp) {
				cudaDeviceSynchronize();
				Tr.mpg[f] += timer(&t_msp);
			}

			MSP.sgl ? cufftExecC2C(cuv[f].plan, (CF*) cuv[f].pmap, (CF*) cuv[f].pmap, -1) :
					cufftExecZ2Z(cuv[f].plan, (CD*) cuv[f].pmap, (CD*) cuv[f].pmap, -1);
			if (Tr.msp) {
				cudaDeviceSynchronize();
				Tr.fft[f] += timer(&t_msp);
			}

			pmapXpr<T2> <<<cuv[f].g_pmap, cuv[f].b_pmap>>>((T2*) cuv[f].pmap,
					((T2*) cuv[f].pr) + (F.slpr[is] - 1) * F.mt * MSP.ntilt[f]);
			if (Tr.msp) {
				cudaDeviceSynchronize();
				Tr.mpr[f] += timer(&t_msp);
			}

			if ((F.sl_opt && i >= F.sl_lo - 1) || i == F.sl_hi - 1) {
				calc_beam<T1, T2><<<cuv[f].g_beam, cuv[f].b_beam>>>(
						(T1*) cuv[f].beam, (T2*) cuv[f].pmap, cuv[f].ib);

				double *beam = F.sl_opt ? F.sl_beam + (i + 1 - F.sl_lo) * F.tottilts * F.mbout
										: F.beam;

				if (MSP.sgl) {
					cudaMemcpyAsync(cuv[f].host_beamf,
							((T1*) cuv[f].beam),
							F.mbout * MSP.ntilt[f] * size_R, cudaMemcpyDeviceToHost);
					for (int j = 0; j < F.mbout; j++) {
						for (int i = 0 ; i < tnt ; i++)
							beam[i + tct1 + j * F.tottilts] =
									(double) cuv[f].host_beamf[i + j * MSP.ntilt[f]];
					}
				} else {
					for (int j = 0; j < F.mbout; j++) {
						cudaMemcpyAsync(beam + tct1 + j * F.tottilts,
								((T1*) cuv[f].beam) + j * MSP.ntilt[f],
								tnt * size_R, cudaMemcpyDeviceToHost);
					}
				}
//				for (int j = 0; j < F.mbout; j++)
//					for (int i = 0 ; i < tnt ; i++) {
//						beam[i + tct1 + j * F.tottilts] = cuv[f].host_beam[i + j * MSP.ntilt[f]];
//						cout << i << " " << j << " " << tct1 << " " << beam[i + tct1 + j * F.tottilts] <<
//								" " << cuv[f].host_beam[i + j * MSP.ntilt[f]] << endl;
//					}

				if (i == F.sl_hi - 1) break;
			}

			if (Tr.msp) {
				cudaDeviceSynchronize();
				t_msp = time_now();
			}
			MSP.sgl ? cufftExecC2C(cuv[f].plan, (CF*) cuv[f].pmap, (CF*) cuv[f].pmap, 1) :
					cufftExecZ2Z(cuv[f].plan, (CD*) cuv[f].pmap, (CD*) cuv[f].pmap, 1);
			if (Tr.msp) {
				cudaDeviceSynchronize();
				Tr.ifft[f] += timer(&t_msp);
			}
		}
	}

	cudaDeviceSynchronize();
//	cudaProfilerStop();
//	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}

extern "C" void cufft_msp_cast(const int f, const int ct1, const int ct2)
{
	if (MSP.sgl){
		for (auto i = 0 ; i < F.mt * F.nslice ; ++i)
			cuv[f].host_pgf[i] = (complex<float>) F.pg[i];
		for (auto i = 0 ; i < 2 * F.tottilts * F.npr ; ++i)
			cuv[f].host_tiltarrayf[i] = (float) F.tiltarray[i];

		cufft_msp<float, float2, float3>(f, ct1, ct2, cuv[f].host_pgf, cuv[f].host_tiltarrayf);
	}
	else
		cufft_msp<double, double2, double3>(f, ct1, ct2, F.pg, F.tiltarray);
}
