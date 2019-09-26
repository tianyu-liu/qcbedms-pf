/*
 * C++ header for all C++ source files of QCBEDMS-PF
 */

#ifndef CPP_COMM_HPP_
#define CPP_COMM_HPP_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <typeinfo>
#include <valarray>

using namespace std;

extern "C" struct {
	complex<double> *pg, *pmap;
	double *deltaz, *xshift, *yshift, *applieddilation, *cell2rpr,
			*beam, *sl_beam, *tiltarray;
	int *res2adr, *ib, *mseq, *slpr;
	int ay, bee, cee, dee, meshx, meshy, mt, npr, nslice, mbout, nump, numl, res2,
			tottilts, sl_hi, sl_lo, ncalcs;
	bool sl_opt;
} F;

extern "C" struct {
	double *ratio;
	char *meth;
	int *dev, *splt, *as_nt, *as_toff, *ntilt;
	int MPI_rank, nhyb, hyb_redi, nfftsg, nfftw, nclfft, ncufft;
	bool sgl;
} MSP;

extern "C" struct {
	double *ch, *pr, *fft, *ifft, *mpg, *mpr;
	bool msp;
} Tr;

extern void fortran_write(const int u, const char *__restrict format, ...);

typedef chrono::time_point<chrono::steady_clock> time_point;
extern time_point time_now();
extern double timer(time_point* t);

extern size_t size_R, size_C;
extern size_t msp_mem(const int f);

#endif /* CPP_COMM_HPP_ */
