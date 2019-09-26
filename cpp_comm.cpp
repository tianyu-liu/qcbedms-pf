/*
 * Common C++ functions of QCBEDMS-PF
 */

#ifdef SINCOS
#	ifndef _GNU_SOURCE
#		define _GNU_SOURCE
#	endif
#	ifdef INTEL_MATH_LIB
#		include <mathimf.h>
#	endif
extern "C" void c_sincos(double *x, double *sn, double *cs)
{
	sincos(*x, sn, cs);
}
#endif

#include "cpp_comm.hpp"


extern "C" void F_C_var_test()
{
	printf("Variables read by C:\n");
	printf("meshx meshy mt = %d %d %d, nslice = %d,\n"
			"mbout = %d, nump numl res2 = %d %d %d, tottilts = %d, \n"
			"nhyb hyb_redi = %d %d, sl_hi sl_lo = %d %d,\n"
			"msp_sgl sl_opt msp_timer = %s %s %s\n",
			F.meshx, F.meshy, F.mt, F.nslice,
			F.mbout, F.nump, F.numl, F.res2, F.tottilts,
			MSP.nhyb, MSP.hyb_redi, F.sl_hi, F.sl_lo,
			MSP.sgl ? "T" : "F", F.sl_opt ? "T" : "F", Tr.msp ? "T" : "F");
	printf("msp_meth = ");
	for (int f = 0 ; f < MSP.nhyb ; f++) {
		char msp_meth[9] = "";
		strncpy(msp_meth, MSP.meth + f * 8, 8);
		printf("%d. '%s', ", f, msp_meth);
	}
	cout << endl;
}

size_t size_R, size_C;

size_t msp_mem(const int f)
{
	size_R = MSP.sgl ? sizeof(float) : sizeof(double);
	size_C = MSP.sgl ? sizeof(complex<float> ) : sizeof(complex<double> );

	size_t pg = F.mt * F.nslice * size_C;
	size_t pr = F.mt * MSP.ntilt[f] * F.nslice * size_C;
	size_t pmap = F.mt * MSP.ntilt[f] * size_C;
	size_t beam = F.res2 * F.mbout * size_R;

	if (strncmp(MSP.meth + f * 8, "fftw", 4) == 0) {
		pmap = F.mt * size_C;
		return pg + pr + pmap * 2;
	}

	return pg + pr + pmap * 2 + beam + MSP.as_nt[f] * sizeof(int);
}

extern "C" void save_as_bmp(int w, int h, double* data, char* name)
{
	unsigned long long filesize = 1078 + w * h;

	unsigned char fileheader[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char infoheader[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0 };
	unsigned char rgbquad[1024] = { 0, 255, 0, 0 }; // map '0' indexed color to Magenta
	unsigned char bmppad[3] = { 0, 0, 0 };

	unsigned char* img = new unsigned char[w * h];
	for (int i = 0 ; i < w * h ; i++)
		img[i] = (unsigned char) data[i];

	// '0' was mapped to Magenta, '1' to '255' are still grayscale
	for (int i = 1 ; i < 256 ; i++) {
		rgbquad[4 * i] = i;
		rgbquad[4 * i + 1] = i;
		rgbquad[4 * i + 2] = i;
		rgbquad[4 * i + 3] = 0;
	}

	fileheader[2] = (unsigned char) (filesize);
	fileheader[3] = (unsigned char) (filesize >> 8);
	fileheader[4] = (unsigned char) (filesize >> 16);
	fileheader[5] = (unsigned char) (filesize >> 24);
	fileheader[10] = (unsigned char) (1078);
	fileheader[11] = (unsigned char) (1078 >> 8);
	fileheader[12] = (unsigned char) (1078 >> 16);
	fileheader[13] = (unsigned char) (1078 >> 24);

	infoheader[4] = (unsigned char) (w);
	infoheader[5] = (unsigned char) (w >> 8);
	infoheader[6] = (unsigned char) (w >> 16);
	infoheader[7] = (unsigned char) (w >> 24);
	infoheader[8] = (unsigned char) (h);
	infoheader[9] = (unsigned char) (h >> 8);
	infoheader[10] = (unsigned char) (h >> 16);
	infoheader[11] = (unsigned char) (h >> 24);
	infoheader[32] = (unsigned char) (256);
	infoheader[33] = (unsigned char) (256 >> 8);
	infoheader[34] = (unsigned char) (256 >> 16);
	infoheader[35] = (unsigned char) (256 >> 24);

	string filename = "";
	filename += name;
	filename += ".bmp";
	FILE* f = fopen(filename.c_str(), "wb");
	fwrite(fileheader, 1, 14, f);
	fwrite(infoheader, 1, 40, f);
	fwrite(rgbquad, 1, 1024, f);
	for (int j = 0 ; j < h ; j++) {
		fwrite(&img[(h - j - 1) * w], 1, w, f);
		fwrite(bmppad, 1, (4 - w % 4) % 4, f);
	}
	fclose(f);

	delete[] img;
}

typedef chrono::time_point<chrono::steady_clock> time_point;

time_point time_now()
{
	return chrono::steady_clock::now();
}

double timer(time_point* t0)
{
	time_point t1 = time_now(), t2 = *t0;
	*t0 = t1;
	return chrono::duration<double>(t1 - t2).count();
}

extern "C" void c2fwrite(int, char*);

void fortran_write(const int u, const char *__restrict format, ...)
{
	if (MSP.MPI_rank > 0) return;
	va_list args;
	va_start(args, format);
	char fmsg[1024];
	vsprintf(fmsg, format, args);
	c2fwrite(13, fmsg);
}

extern "C" void msp_child_exec_bridge_f(const int f);

extern "C" void msp_cppthread_exec(const int nf, const int offset)
{
	thread *hybrid = new thread[nf];
	for (int f = 0 ; f < nf ; f++) {
		hybrid[f] = thread(msp_child_exec_bridge_f, f + 1 + offset);
//		hybrid[f].join();
	}

	for (int f = 0 ; f < nf ; f++) {
		hybrid[f].join();
	}
	delete[] hybrid;
}


