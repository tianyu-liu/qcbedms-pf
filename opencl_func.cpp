/*
 * OpenCL functions of QCBEDMS-PF
 */

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <clFFT.h>
#include "cpp_comm.hpp"

using namespace std;

struct clfft_vars {
	cl_context ctx;
	cl_command_queue q;
	cl_mem pmap, ptmp, *pg, *pr, beam, ib, tiltarray;
	cl_program program;
	cl_kernel propg, pad0oshift, fill_pmap, fill_beam, beam_out;
	clfftPlanHandle *plans_f, plan_b;
	size_t G_pmap, L_pmap, G_beam_out[2] = { 1, 1 }, L_beam_out[2] = { 1, 1 },
			G_propg[3] = { 1, 1, 1 }, L_propg[3] = { 1, 1, 1 }, G_pad, L_pad;
	complex<float> *host_pgf;
	float *host_tiltarrayf, *host_beamf;
}*clv;

extern "C"
{
void cl_dbg(string msg, cl_int err)
{
	if(err != 0)
		printf("%s; error code = %d\n", msg.c_str(), err);
}

void clfft_init(int *count)
{
	clv = new clfft_vars[MSP.nhyb];
	
/* Initialise clFFT */
	clfftSetupData fftSetup;
	cl_dbg("clfftInitSetupData", clfftInitSetupData(&fftSetup));
	cl_dbg("clfftSetup", clfftSetup(&fftSetup));
}

void clfft_term()
{
	delete[] clv, clv = NULL;
	clfftTeardown();
}

cl_device_id OpenCLGetGPU(int idev, int *gpu_count)
{
	cl_device_id dev_id;
	cl_uint num_plat;
	clGetPlatformIDs(0, NULL, &num_plat);
	cl_platform_id *plats = new cl_platform_id[num_plat];
	clGetPlatformIDs(num_plat, plats, NULL);

	cl_uint num_dev = 0;
	cl_uint idev_gb = 0;
	char plat_name[128];
	char device_name[128];
	char vend_name[128];
	int count = 0;

	if (idev == -1)
		printf("OpenCL GPU devices:\n");

	for (cl_uint i = 0 ; i < num_plat ; i++, num_dev = 0) {
		clGetPlatformInfo(plats[i], CL_PLATFORM_NAME,
				sizeof(plat_name), plat_name, NULL);
		clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_dev);
		count += num_dev;

		cl_device_id* devices = new cl_device_id[num_dev];
		clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, num_dev, devices, NULL);

		for (cl_uint j = 0 ; j < num_dev ; j++, idev_gb++) {
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME,
					sizeof(device_name), device_name, NULL);
			clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR,
					sizeof(vend_name), vend_name, NULL);
			if (idev == -1)
				printf("  %i. %s, Vendor: %s, Platform: %s.\n",
						idev_gb, device_name, vend_name, plat_name);
			if (idev == (int) idev_gb) dev_id = devices[j];
		}

		delete[] devices, devices = NULL;
	}

	delete[] plats, plats = NULL;

	if (idev == -1)
		exit(0);
	else if (idev == -2) {
		*gpu_count = count;
		return 0;
	}

	cl_platform_id plat;
	clGetDeviceInfo(dev_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
	clGetDeviceInfo(dev_id, CL_DEVICE_VENDOR, sizeof(vend_name), vend_name, NULL);
	clGetDeviceInfo(dev_id, CL_DEVICE_PLATFORM, sizeof(plat), &plat, NULL);
	clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(plat_name), plat_name, NULL);
	fortran_write(13, "  Device: %d. %s, Vendor: %s, Platform/Driver: %s. \n",
			idev, device_name, vend_name, plat_name);

	return dev_id;
}

void clfft_child_init(const int f)
{
	cl_int err;
	cl_device_id dev_id = OpenCLGetGPU(MSP.dev[f], NULL);
	cl_platform_id plat;
	err = clGetDeviceInfo(dev_id, CL_DEVICE_PLATFORM, sizeof(plat), &plat, NULL);

/* Check double precision support on GPU */
	fortran_write(13, "  clFFT precision: %s.\n", MSP.sgl ? "single" : "double");
	cl_device_fp_config fp;
	err = clGetDeviceInfo(dev_id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp), &fp, NULL);
	if (fp == 0 && !MSP.sgl) {
		fortran_write(13, "\nDouble floating point precision not supported by the device!");
		cout << "Double floating-point precision not supported by the device!" << endl;
		exit(-1);
	}

/* Calculate GPU memory usage */
	float MB = 1.0 / 1024 / 1024;
	size_t total_gm, max_alloc, work_gm;
	cl_int mem_err1 = clGetDeviceInfo(dev_id, CL_DEVICE_GLOBAL_MEM_SIZE,
			sizeof(total_gm), &total_gm, NULL);
	cl_int mem_err2 = clGetDeviceInfo(dev_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
			sizeof(max_alloc), &max_alloc, NULL);
	if (MSP.splt[f] == 0) {
		MSP.splt[f] = 1, MSP.ntilt[f] = MSP.as_nt[f];
		/* Apply total memory and max allocation criterion */
		while (msp_mem(f) > total_gm && F.mt * MSP.ntilt[f] * size_C > max_alloc
				&& MSP.splt[f] < MSP.as_nt[f])
			++MSP.splt[f], MSP.ntilt[f] = (int) ceil((float) MSP.as_nt[f] / MSP.splt[f]);
		fortran_write(13, "  (AUTO)");
	}
	else
		MSP.ntilt[f] = (int) ceil((float) MSP.as_nt[f] / MSP.splt[f]);

	fortran_write(13, "  Paralleling %d/%d tilts; will loop %d time(s).\n",
			MSP.ntilt[f], MSP.as_nt[f], MSP.splt[f]);
	work_gm = msp_mem(f);
	fortran_write(13, "  Total GPU memory: %g MB; clFFT will use about %g MB.\n",
			total_gm * MB, work_gm * MB);
	fortran_write(13, "  The max GPU buffer is %g MB (Device's limit is %g MB).\n",
			F.mt * MSP.ntilt[f] * size_C * MB, max_alloc * MB);

	if (work_gm > total_gm || F.mt * MSP.ntilt[f] * size_C > max_alloc) {
		if (F.ncalcs < 1) {
			fortran_write(13, "\nNOT enough GPU memory OR the max array excesses GPU's limit! "
					"\nPlease reduce paralleled tilts manually.\n");
			cout << "NOT enough GPU memory OR the max GPU array excesses device's limit! "
					"Please reduce paralleled tilts manually." << endl;
			exit(-1);
		} else if (mem_err1 != CL_SUCCESS || mem_err2 != CL_SUCCESS) {
			fortran_write(13, "\nWARNING: \"OpenCL get memory info\" failed "
					"during refinement! Ignored and continue.\n");
		}
	}
	if (work_gm > total_gm * 0.5)
		fortran_write(13, "  GPU memory may be intensive; reduce paralleled "
				"tilts manually if any error occurs.\n");

/* Initiate memory objects */
	err = clGetDeviceInfo(dev_id, CL_DEVICE_PLATFORM, sizeof(plat), &plat, NULL);
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	props[1] = (cl_context_properties) plat;
	clv[f].ctx = clCreateContext(props, 1, &dev_id, NULL, NULL, &err);
	cl_dbg("clCreateContext", err);
	clv[f].q = clCreateCommandQueue(clv[f].ctx, dev_id, 0, &err);
	cl_dbg("clCreateCommandQueue", err);

	if(MSP.sgl) {
		clv[f].host_pgf = new complex<float>[F.mt * F.nslice];
		clv[f].host_tiltarrayf = new float[2 * F.tottilts * F.npr];
		clv[f].host_beamf = new float[F.mbout * MSP.ntilt[f] * size_R];
	}
	clv[f].pg = new cl_mem[F.nslice];
	clv[f].pr = new cl_mem[F.npr];
	clv[f].tiltarray = clCreateBuffer(clv[f].ctx, CL_MEM_READ_ONLY,// | CL_MEM_HOST_WRITE_ONLY,
			2 * MSP.ntilt[f] * size_R, NULL, &err);
	for (int i = 0 ; i < F.nslice ; i++)
		clv[f].pg[i] = clCreateBuffer(clv[f].ctx, CL_MEM_READ_ONLY,// | CL_MEM_HOST_WRITE_ONLY,
				F.mt * size_C, NULL, &err);
	for (int i = 0 ; i < F.npr ; i++)
		clv[f].pr[i] = clCreateBuffer(clv[f].ctx, CL_MEM_READ_ONLY,// | CL_MEM_HOST_WRITE_ONLY,
				F.mt * MSP.ntilt[f] * size_C, NULL, &err);
	clv[f].pmap = clCreateBuffer(clv[f].ctx, CL_MEM_READ_WRITE,// | CL_MEM_HOST_WRITE_ONLY,
			F.mt * MSP.ntilt[f] * size_C, NULL, &err);
	clv[f].ptmp = clCreateBuffer(clv[f].ctx, CL_MEM_READ_WRITE,// | CL_MEM_HOST_NO_ACCESS,
			F.mt * MSP.ntilt[f] * size_C, NULL, &err);
	clv[f].beam = clCreateBuffer(clv[f].ctx, CL_MEM_READ_WRITE,// | CL_MEM_HOST_READ_ONLY,
			F.mbout * MSP.ntilt[f] * size_R, NULL, &err);
	clv[f].ib = clCreateBuffer(clv[f].ctx,
			CL_MEM_READ_ONLY,// | CL_MEM_HOST_WRITE_ONLY| CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
			F.mbout * sizeof(int), NULL, &err);
	// CL_MEM_COPY_HOST_PTR appears to cause memory leak on Nvidia/Linux.
	// CL_MEM_HOST_WRITE_ONLY on macOS GPUs would fail to allocate buffers larger than 128 MiB.
	// All unnecessary flags are removed, even they may benefit the performance.
	err = clEnqueueWriteBuffer(clv[f].q, clv[f].ib, CL_FALSE,
			0, F.mbout * sizeof(int), F.ib, 0, NULL, NULL);

/* Plan clFFT backward transform */
	size_t clfftLengths[2] = { (size_t) F.meshx, (size_t) F.meshy };
	err = clfftCreateDefaultPlan(&clv[f].plan_b, clv[f].ctx, CLFFT_2D, clfftLengths);
	err = clfftSetPlanBatchSize(clv[f].plan_b, MSP.ntilt[f]);
	err = clfftSetPlanDistance(clv[f].plan_b, F.mt, F.mt);
	err = clfftSetPlanPrecision(clv[f].plan_b, MSP.sgl ? CLFFT_SINGLE : CLFFT_DOUBLE);
	err = clfftSetLayout(clv[f].plan_b, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	err = clfftSetResultLocation(clv[f].plan_b, CLFFT_INPLACE);
	err = clfftBakePlan(clv[f].plan_b, 1, &clv[f].q, NULL, NULL);

/* Set precision & global constants for kernels */
	string p1 = MSP.sgl ? "float" : "double";
	string p2 = MSP.sgl ? "float2" : "double2";
	string p4 = MSP.sgl ? "float4" : "double4";
	string mx = to_string(F.meshx / 2), my = to_string(F.meshy / 2), mthf = to_string(F.mt / 2),
			meshx = to_string(F.meshx), meshy = to_string(F.meshy), mt = to_string(F.mt),
			ntilt = to_string(MSP.ntilt[f]);

/* Setup clFFT callbacks for forward transform */
	string preXpg_str = ""
""+p2+" preXpg(__global "+p2+"* restrict pmap, const uint offset, 		\n"
"		const __global "+p2+"* restrict pg)								\n"
"{																		\n"
"	"+p2+" pgi = *(pg + offset%"+mt+"), pmapi = *(pmap+offset);			\n"
//"	if(offset<1) {														\n"
//"		printf(\"pmap %i %.15g %.15g\\n\",offset,pmapi.x,pmapi.y);		\n"
//"		printf(\"pg %i %.15g %.15g\\n\",offset,pgi.x,pgi.y);			\n"
//"		"+p2+" m= (pmapi.x * pgi.x - pmapi.y * pgi.y, 					\n"
//"					pmapi.x * pgi.y + pmapi.y * pgi.x);					\n"
//"		printf(\"pmap %i %.15g %.15g\\n\",offset,m.x,m.y);				\n"
//"	}																	\n"
"	return ("+p2+") (pmapi.x * pgi.x - pmapi.y * pgi.y, 				\n"
"						pmapi.x * pgi.y + pmapi.y * pgi.x);				\n"
"}\n";

	string postXpr_str = ""
"void postXpr(__global "+p2+"* restrict pmap, const uint offset, 	\n"
"		const __global "+p2+"* restrict pr, const "+p2+" fftO)		\n"
"{																	\n"
"	"+p2+" pri = *(pr + offset);									\n"
//"	if(offset<5) printf(\"pr %i %.15g %.15g\\n\",offset,pri.x,pri.y);\n"
"	*(pmap + offset) = ("+p2+") (fftO.x * pri.x - fftO.y * pri.y,	\n"
"								 fftO.x * pri.y + fftO.y * pri.x);	\n"
"}";

/* Plan clFFT forward transform */
	clv[f].plans_f = new clfftPlanHandle[F.nslice];
	for (int i = 0 ; i < F.nslice ; i++) {
		err = clfftCopyPlan(&clv[f].plans_f[i], clv[f].ctx, clv[f].plan_b);
		err = clfftSetPlanCallback(clv[f].plans_f[i], "preXpg", preXpg_str.c_str(),
				0, PRECALLBACK, &clv[f].pg[i], 1);
		err = clfftSetPlanCallback(clv[f].plans_f[i], "postXpr", postXpr_str.c_str(),
				0, POSTCALLBACK, &clv[f].pr[F.slpr[i] - 1], 1);
		err = clfftBakePlan(clv[f].plans_f[i], 1, &clv[f].q, NULL, NULL);
	}

/* Setup other kernels */
	string program_str = ""
"__kernel void propg(const "+p4+" xyz, const "+p4+" c2r, 							\n"
"		const __global "+p2+"* restrict tiltarray, __global "+p2+"* restrict pr)	\n"
"{																					\n"
"	const int x = get_global_id(1), y = get_global_id(2), t = get_group_id(0);		\n"
"	const int h = x - "+mx+", k = y -  "+my+";										\n"
"	const "+p2+" tilt = tiltarray[t];												\n"
"	"+p1+" a1, b1, sg, sn, cs;														\n"
"	a1 = tilt.x + h, b1 = tilt.y + k;												\n"
"	sg = a1 * h * c2r.x + b1 * k * c2r.y + (a1 * k + b1 * h) * c2r.z;				\n"
"	sn = sincos(6.283185307179586477 * (sg * xyz.z + h * xyz.x + k * xyz.y), &cs);	\n"
"	pr[x + y *  "+meshx+" + t *  "+mt+"] = ("+p2+") {cs, sn};						\n"
//"	if (x==0&&y==0)																	\n"
//"		printf(\"%d %g %g\\n\",t,tilt.x,tilt.y);									\n"
//"	if (t == 0 && x==0&&y==0)														\n"
//"		printf(\"%g %g %g %g %g %g sizeof(U)=%i\\n\", 								\n"
//"				xyz.x,xyz.y,xyz.z,c2r.x,c2r.y,c2r.z,sizeof(xyz));					\n"
//"	if (t == 1 && x!=0&&y!=0)														\n"
//"		printf(\"%i.%i %i %i %i %g %g\\n\", x,y, h,k,x+y*"+meshx+",cs,sn);			\n"
"}\n"

"__kernel void pad0oshift(__global "+p2+"* restrict pr)									\n"
"{																						\n"
"	int tmt = get_global_id(0);															\n"
"	if (tmt < "+ntilt+") {																\n"
"		tmt *=  "+mt+";																	\n"
// PAD0
"		"+p2+" m = {0, 0};																\n"
"		for (int i = tmt ; i <  "+meshx+" + tmt ; ++i)									\n"
"			pr[i] = m;																	\n"
"		for (int j = "+meshx+" + tmt ; j < "+mt+" + tmt ; j += "+meshx+")				\n"
"			pr[j] = m;																	\n"
// OSHIFT
"		for (int j = tmt ; j < "+mthf+" - "+meshx+" + tmt + 1 ; j += "+meshx+")			\n"
"			for (int i = j, i1 = i +  "+mx+" + "+mthf+" ; i < "+mx+" + j ; ++i, ++i1)	\n"
"				m = pr[i], pr[i] = pr[i1], pr[i1] = m;									\n"
"		for (int j = "+mthf+" + tmt ; j < "+mt+" - "+meshx+" + tmt + 1 ; j += "+meshx+")\n"
"			for (int i = j, i1 = i + "+mx+" - "+mthf+" ; i < "+mx+" + j ; ++i, ++i1)	\n"
"				m = pr[i], pr[i] = pr[i1], pr[i1] = m;									\n"
//"		if(tmt/"+mt+"==0)																\n"
//"		for (int i = 0 ; i < "+mt+" ; ++i)												\n"
//"			printf(\"%i-%i. %g %g\\n\",tmt/"+mt+",i,pr[tmt+i].x,pr[tmt+i].y);			\n"
"	}																					\n"
"}\n"

"__kernel void fill_pmap(__global "+p2+"* pmap)	\n"
"{												\n"
"	int i = get_global_id(0);					\n"
"	pmap[i] = ("+p2+") {1./"+mt+", 0};			\n"
"}\n"

"__kernel void beam_out(__global "+p1+"* restrict beam,							\n"
"		const __global "+p2+"* restrict pmap, const __global int* restrict ib)	\n"
"{																				\n"
"	int i = get_global_id(0), j = get_group_id(1);								\n"
"	if (i < "+ntilt+") {														\n"
"		"+p2+" m = pmap[i * "+mt+" + ib[j] - 1];								\n"
"		beam[i + j * "+ntilt+"] = ("+p1+") m.x * m.x + m.y * m.y;				\n"
//"		if(i<5) printf(\"pmap %i %.15g %.15g\\n\",i,m.x,m.y);					\n"
//"		printf(\"beam: %d %d %d %g\\n\",j,i,i + j * "+ntilt+",beam[i + j * "+ntilt+"]);	\n"
"	}\n"
"}\n";
	const char* program_cstr = program_str.c_str();

	clv[f].program = clCreateProgramWithSource(clv[f].ctx, 1, &program_cstr, NULL, &err);
	err = clBuildProgram(clv[f].program, 1, &dev_id, NULL, NULL, NULL);
	cl_build_status build_status;
	clGetProgramBuildInfo(clv[f].program, dev_id, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
	if (build_status != CL_BUILD_SUCCESS) {
		size_t build_log_size;
		clGetProgramBuildInfo(clv[f].program, dev_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
		char* build_log = (char*) malloc(build_log_size);
		clGetProgramBuildInfo(clv[f].program, dev_id, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
		printf("%s\n", build_log);
		free(build_log);
	}

	clv[f].propg = clCreateKernel(clv[f].program, "propg", &err);

	clv[f].pad0oshift = clCreateKernel(clv[f].program, "pad0oshift", &err);

	clv[f].fill_pmap = clCreateKernel(clv[f].program, "fill_pmap", &err);
	err = clSetKernelArg(clv[f].fill_pmap, 0, sizeof(cl_mem), &clv[f].pmap);

	clv[f].beam_out = clCreateKernel(clv[f].program, "beam_out", &err);
	err = clSetKernelArg(clv[f].beam_out, 0, sizeof(cl_mem), &clv[f].beam);
	err = clSetKernelArg(clv[f].beam_out, 1, sizeof(cl_mem), &clv[f].pmap);
	err = clSetKernelArg(clv[f].beam_out, 2, sizeof(cl_mem), &clv[f].ib);

/* Setup GPU multi-thread job distribution */
	size_t maxWorkSize[3], maxWorkGroup;
	clGetDeviceInfo(dev_id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
			sizeof(maxWorkSize), maxWorkSize, NULL);
	clGetDeviceInfo(dev_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
			sizeof(maxWorkGroup), &maxWorkGroup, NULL);

	clv[f].L_propg[1] = F.meshx, clv[f].L_propg[2] = F.meshy;
	while (clv[f].L_propg[2] > min(maxWorkSize[2], maxWorkGroup))
		clv[f].L_propg[2] /= 2;
	while (clv[f].L_propg[1] * clv[f].L_propg[2] > maxWorkGroup)
		clv[f].L_propg[1] /= 2;
	clv[f].G_propg[0] = MSP.ntilt[f], clv[f].G_propg[1] = F.meshx, clv[f].G_propg[2] = F.meshy;

	clv[f].L_pad = MSP.ntilt[f];
	if (clv[f].L_pad > min(maxWorkGroup, maxWorkSize[0]))
		clv[f].L_pad = min(maxWorkGroup, maxWorkSize[0]);
	clv[f].G_pad = ceil(1.0 * MSP.ntilt[f] / clv[f].L_pad) * clv[f].L_pad;

	clv[f].L_pmap = F.mt, clv[f].G_pmap = F.mt * MSP.ntilt[f];
	while (clv[f].L_pmap > min(maxWorkSize[2], maxWorkGroup))
		clv[f].L_pmap /= 2;

	clv[f].L_beam_out[0] = MSP.ntilt[f];
	if (clv[f].L_beam_out[0] > maxWorkGroup)
		clv[f].L_beam_out[0] = maxWorkGroup;
	clv[f].G_beam_out[0] = ceil(1.0 * MSP.ntilt[f] / clv[f].L_beam_out[0]) * clv[f].L_beam_out[0];
	clv[f].G_beam_out[1] = F.mbout;

	fortran_write(13, "  Device's max total size of a work group = %d & per dimension = (%d, %d, %d)\n",
			maxWorkGroup, maxWorkSize[0], maxWorkSize[1], maxWorkSize[2]);
	fortran_write(13, "  Kernels' work group sizes will be:\n");
	fortran_write(13, "    propg      - local = (%d, %d, %d),  global = (%d, %d, %d);\n",
			clv[f].L_propg[0], clv[f].L_propg[1], clv[f].L_propg[2],
			clv[f].G_propg[0], clv[f].G_propg[1], clv[f].G_propg[2]);
	fortran_write(13, "    pad0oshift - local = (%d),  global = (%d);\n", clv[f].L_pad, clv[f].G_pad);
	fortran_write(13, "    pmap       - local = (%d),  global = (%d);\n", clv[f].L_pmap, clv[f].G_pmap);
	fortran_write(13, "    beam_out   - local = (%d, %d),  global = (%d, %d).\n",
			clv[f].L_beam_out[0], clv[f].L_beam_out[1], clv[f].G_beam_out[0], clv[f].G_beam_out[1]);
}

void clfft_child_term(const int f)
{
	string sf = " gpu=" + to_string(f);
	if(MSP.sgl) {
		delete[] clv[f].host_pgf;
		delete[] clv[f].host_tiltarrayf;
		delete[] clv[f].host_beamf;
	}
	for (int i = 0 ; i < F.nslice ; i++) {
		cl_dbg("clReleaseMemObject pg" + sf, clReleaseMemObject(clv[f].pg[i]));
		cl_dbg("clfftDestroyPlan plans_f" + sf, clfftDestroyPlan(&clv[f].plans_f[i]));
	}
	for (int i = 0 ; i < F.npr ; i++)
		cl_dbg("clReleaseMemObject pr" + sf, clReleaseMemObject(clv[f].pr[i]));
	delete[] clv[f].pg;
	delete[] clv[f].pr;
	delete[] clv[f].plans_f;
	cl_dbg("clfftDestroyPlan plan_b" + sf, clfftDestroyPlan(&clv[f].plan_b));
	cl_dbg("clReleaseMemObject ptmp" + sf, clReleaseMemObject(clv[f].ptmp));
	cl_dbg("clReleaseMemObject pmap" + sf, clReleaseMemObject(clv[f].pmap));
	cl_dbg("clReleaseMemObject beam" + sf, clReleaseMemObject(clv[f].tiltarray));
	cl_dbg("clReleaseMemObject beam" + sf, clReleaseMemObject(clv[f].beam));
	cl_dbg("clReleaseMemObject ib" + sf, clReleaseMemObject(clv[f].ib));
	cl_dbg("clReleaseKernel propg" + sf, clReleaseKernel(clv[f].propg));
	cl_dbg("clReleaseKernel pad0oshift" + sf, clReleaseKernel(clv[f].pad0oshift));
	cl_dbg("clReleaseKernel fill_pmap" + sf, clReleaseKernel(clv[f].fill_pmap));
	cl_dbg("clReleaseKernel beam_out" + sf, clReleaseKernel(clv[f].beam_out));
	cl_dbg("clReleaseProgram" + sf, clReleaseProgram(clv[f].program));
	cl_dbg("clReleaseCommandQueue" + sf, clReleaseCommandQueue(clv[f].q));
	cl_dbg("clReleaseContext" + sf, clReleaseContext(clv[f].ctx));
}
}

template<typename T, typename U>
void clfft_msp(const int f, const int ct1, const int ct2, complex<T> *host_pg, T *host_tiltarray)
{
	time_point t_msp;
	for (int i = 0 ; i < F.nslice ; i++)
		cl_dbg("Write pg", clEnqueueWriteBuffer(clv[f].q, clv[f].pg[i], CL_FALSE,
				0, F.mt * size_C, host_pg + i * F.mt, 0, NULL, NULL));

	for (int isplt = 0 ; isplt < MSP.splt[f] ; ++isplt) {
		int tct1 = isplt * MSP.ntilt[f] + ct1;
		int tnt = min(MSP.ntilt[f], ct2 - tct1 + 1);
//		printf("%d %d %d %lu\n",tct1,ct2,ct2 - tct1 + 1,copy_size/size_R);

		if (Tr.msp) {
			clFinish(clv[f].q);
			t_msp = time_now();
		}
		for (int i = 0 ; i < F.npr ; i++) {
			cl_dbg("Write tiltarray", clEnqueueWriteBuffer(clv[f].q, clv[f].tiltarray, CL_FALSE,
					0, 2 * tnt * size_R, host_tiltarray + 2 * (tct1 + i * F.tottilts), 0, NULL, NULL));

			U xyz = { (T) F.xshift[i], (T) F.yshift[i], (T) (F.deltaz[i] * F.applieddilation[i]) };
			U c2r = { (T) F.cell2rpr[3 * i], (T) F.cell2rpr[3 * i + 1], (T) F.cell2rpr[3 * i + 2] };
//			printf("%g %g %g %g %g %g\n", xyz.s[0],xyz.s[1],xyz.s[2],c2r.s[0],c2r.s[1],c2r.s[2]);
			cl_dbg("propg set xyz", clSetKernelArg(clv[f].propg, 0, sizeof(U), &xyz));
			cl_dbg("propg set c2r", clSetKernelArg(clv[f].propg, 1, sizeof(U), &c2r));
			cl_dbg("propg set tilt", clSetKernelArg(clv[f].propg, 2, sizeof(cl_mem), &clv[f].tiltarray));
			cl_dbg("propg set pr", clSetKernelArg(clv[f].propg, 3, sizeof(cl_mem), &clv[f].pr[i]));
			cl_dbg("propg enqueue", clEnqueueNDRangeKernel(clv[f].q, clv[f].propg, 3, 0,
					clv[f].G_propg, clv[f].L_propg, 0, NULL, NULL));

			cl_dbg("pad0oshift set pr", clSetKernelArg(clv[f].pad0oshift, 0, sizeof(cl_mem), &clv[f].pr[i]));
			cl_dbg("pad0oshift enqueue", clEnqueueNDRangeKernel(clv[f].q, clv[f].pad0oshift, 1, 0,
					&clv[f].G_pad, &clv[f].L_pad, 0, NULL, NULL));
		}
		if (Tr.msp) {
			clFinish(clv[f].q);
			Tr.pr[f] += timer(&t_msp);
		}

		// clEnqueueFillBuffer is buggy on Mac; a kernel is required to fill clv[f].pmap
		//	complex<T> cmplx1 = { (T) 1.0 / F.mt, 0 };
		//	clEnqueueFillBuffer(clv[f].q, clv[f].pmap, &cmplx1,
		//			sizeof(cmplx1), 0, Size.pmap, 0, NULL, NULL);
		clEnqueueNDRangeKernel(clv[f].q, clv[f].fill_pmap, 1, 0,
				&clv[f].G_pmap, &clv[f].L_pmap, 0, NULL, NULL);

		if (Tr.msp) {
			clFinish(clv[f].q);
			t_msp = time_now();
		}

		for (int i = 0 ; i < F.sl_hi ; i++) {
			cl_dbg("clfftEnqueueTransform FFT", clfftEnqueueTransform(clv[f].plans_f[F.mseq[i] - 1],
					CLFFT_FORWARD, 1, &clv[f].q, 0, NULL, NULL, &clv[f].pmap, NULL, clv[f].ptmp));
			if (Tr.msp) {
				clFinish(clv[f].q);
				Tr.fft[f] += timer(&t_msp);
			}

			if ((F.sl_opt && i >= F.sl_lo - 1) || i == F.sl_hi - 1) {
				cl_dbg("clEnqueueNDRangeKernel beam_out", clEnqueueNDRangeKernel(clv[f].q,
						clv[f].beam_out, 2, 0, clv[f].G_beam_out, clv[f].L_beam_out, 0, NULL, NULL));

				double* beam = F.sl_opt ? F.sl_beam + (i + 1 - F.sl_lo) * F.tottilts * F.mbout
										: F.beam;

				if (MSP.sgl) {
					cl_dbg("clEnqueueReadBuffer beam",
							clEnqueueReadBuffer(clv[f].q, clv[f].beam, CL_TRUE,
									0, F.mbout * MSP.ntilt[f] * size_R, clv[f].host_beamf,
									0, NULL, NULL));
					for (int j = 0 ; j < F.mbout ; j++)
						for (int i = 0 ; i < tnt ; i++)
							beam[i + tct1 + j * F.tottilts] =
									(double) clv[f].host_beamf[i + j * MSP.ntilt[f]];
				}
				else {
					for (int j = 0 ; j < F.mbout ; j++) {
						cl_dbg("clEnqueueReadBuffer beam",
								clEnqueueReadBuffer(clv[f].q, clv[f].beam, CL_FALSE,
										j * MSP.ntilt[f] * size_R, tnt * size_R,
										beam + tct1 + j * F.tottilts, 0, NULL, NULL));
//						printf("%d %d %d %d %d\n", j, j * MSP.ntilt[f], tct1, tnt, tct1 + j * F.tottilts);
					}
				}

				if (i == F.sl_hi - 1) break;
			}

			if (Tr.msp) {
				clFinish(clv[f].q);
				t_msp = time_now();
			}
			cl_dbg("clfftEnqueueTransform IFFT", clfftEnqueueTransform(clv[f].plan_b,
					CLFFT_BACKWARD, 1, &clv[f].q, 0, NULL, NULL, &clv[f].pmap, NULL, clv[f].ptmp));
			if (Tr.msp) {
				clFinish(clv[f].q);
				Tr.ifft[f] += timer(&t_msp);
			}
		}
	}

	clFinish(clv[f].q);
}

extern "C" void clfft_msp_cast(const int f, const int ct1, const int ct2)
{
	if (MSP.sgl){
		for (auto i = 0 ; i < F.mt * F.nslice ; ++i)
			clv[f].host_pgf[i] = (complex<float>) F.pg[i];
		for (auto i = 0 ; i < 2 * F.tottilts * F.npr ; ++i)
			clv[f].host_tiltarrayf[i] = (float) F.tiltarray[i];

		clfft_msp<float, cl_float4>(f, ct1, ct2, clv[f].host_pgf, clv[f].host_tiltarrayf);
	}
	else
		clfft_msp<double, cl_double4>(f, ct1, ct2, F.pg, F.tiltarray);
}
