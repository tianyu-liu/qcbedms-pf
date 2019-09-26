/*
 * FFTW functions of QCBEDMS-PF
 */
#include "cpp_comm.hpp"
#include <fftw3.h>

using namespace std;
typedef fftw_complex CD;
typedef fftwf_complex CF;

struct fftw_vars {
	CD *pr, *pmap;
	CF *prf, *pmapf;
	fftw_plan msf, msb;
	fftwf_plan msff, msbf;
	time_point t_msp;
} *fwv;

fftw_plan p1f, p1b;

extern "C"
{
void fftw_init()
{
	fwv = new fftw_vars[MSP.nhyb];
	F.pg = reinterpret_cast<complex<double>*>(fftw_alloc_complex(F.mt * F.nslice));
	F.pmap = reinterpret_cast<complex<double>*>(fftw_alloc_complex(F.mt));
	p1f = fftw_plan_dft_2d(F.meshy, F.meshx,
			reinterpret_cast<CD*>(F.pmap), reinterpret_cast<CD*>(F.pmap),
			FFTW_FORWARD, FFTW_PATIENT);
	p1b = fftw_plan_dft_2d(F.meshy, F.meshx,
			reinterpret_cast<CD*>(F.pmap), reinterpret_cast<CD*>(F.pmap),
			FFTW_BACKWARD, FFTW_PATIENT);
}

void fftw_term()
{
	delete[] fwv;
	fftw_free(reinterpret_cast<CD*>(F.pg));
	fftw_free(reinterpret_cast<CD*>(F.pmap));
	fftw_destroy_plan(p1f);
	fftw_destroy_plan(p1b);
}

void fftw_one(const int dir)
{
	dir == -1 ? fftw_execute(p1f) : fftw_execute(p1b);
}

void fftw_child_init(const int f)
{
	MSP.ntilt[f] = 1;
	fortran_write(13, "  FFTW thread # %d will calculate %d tilts; "
			"using at least %g MB\n",
			MSP.dev[f], MSP.as_nt[f], msp_mem(f) / 1024.0 / 1024);

	if(MSP.sgl) {
		fwv[f].prf = fftwf_alloc_complex(F.mt * F.npr);
		fwv[f].pmapf = fftwf_alloc_complex(F.mt);
		fwv[f].msff = fftwf_plan_dft_2d(F.meshy, F.meshx,
				fwv[f].pmapf, fwv[f].pmapf, FFTW_FORWARD, FFTW_PATIENT);
		fwv[f].msbf = fftwf_plan_dft_2d(F.meshy, F.meshx,
				fwv[f].pmapf, fwv[f].pmapf, FFTW_BACKWARD, FFTW_PATIENT);
	} else {
		fwv[f].pr = fftw_alloc_complex(F.mt * F.npr);
		fwv[f].pmap = fftw_alloc_complex(F.mt);
		fwv[f].msf  = fftw_plan_dft_2d(F.meshy, F.meshx,
				fwv[f].pmap, fwv[f].pmap,
				FFTW_FORWARD, FFTW_PATIENT);
		fwv[f].msb = fftw_plan_dft_2d(F.meshy, F.meshx,
				fwv[f].pmap, fwv[f].pmap,
				FFTW_BACKWARD, FFTW_PATIENT);
	}
}

void fftw_child_term(const int f)
{
	if(MSP.sgl) {
		fftwf_free(fwv[f].prf);
		fftwf_free(fwv[f].pmapf);
		fftwf_destroy_plan(fwv[f].msff);
		fftwf_destroy_plan(fwv[f].msbf);
	} else {
		fftw_free(fwv[f].pr);
		fftw_free(fwv[f].pmap);
		fftw_destroy_plan(fwv[f].msf);
		fftw_destroy_plan(fwv[f].msb);
	}
}

void fftw_time_now(const int f)
{
	fwv[f].t_msp = time_now();
}

double fftw_timer(const int f)
{
	return timer(&fwv[f].t_msp);
}

void fftw_fft(const int f, const int dir)
{
	if(MSP.sgl) {
		if(dir == -1)
			fftwf_execute(fwv[f].msff);
		else
			fftwf_execute(fwv[f].msbf);
	} else {
		if (dir == -1)
			fftw_execute(fwv[f].msf);
		else
			fftw_execute(fwv[f].msb);
	}
}

void fftw_msp_f(const int f, const int ct1, const int ct2,
		complex<double> *pg, CD *pr, CD *pmap);

void fftw_msp_f_bridge(const int f, const int ct1, const int ct2)
{
	fftw_msp_f(f + 1, ct1 + 1, ct2 + 1, F.pg, fwv[f].pr, fwv[f].pmap);
}
}
