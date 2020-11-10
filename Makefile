
#  make        ... generate executable
#  make clean  ... delete unnecessary files

# EXE ........ name of the resulting executable
# FC ......... FORTRAN (and linking) compiler
# CXX ........ C++ compiler
# OPT ........ optimisation flags applying to C++/FORTRAN
# FGEN ....... FORTRAN code generation flags
# CXXGEN ..... C/C++ code generation flags
# LDFLAGS .... linker flags & libraries needed to build the executable

FC	= ifort #-check all -warn -nogen-interfaces
CXX	= icpc
OPT = -O3 -ipo -march=core-avx2 -fp-model fast=2 #-static-intel
#OPT	= -O0 -g -fp-model strict -traceback -debug extended

FC	= gfortran
CXX	= g++
OPT	= -Ofast -march=native #-fno-fast-math
#OPT	= -O0 #-g -fcheck=all -Wall

 #-ffpe-trap=underflow,denormal,overflow,zero,invalid  #-fcheck=all -Wall
 #-fsanitize=address -fno-omit-frame-pointer
# ↑ ifort/icpc   OPT: -O3 -fp-model strict -march=core-avx2 -ipo
# ↑ gfortran/g++ OPT: -Ofast -march=native 

# ↓ Linking compiler; Fortran compilers are recommended 
LC	= $(FC)
FGEN	=
CXXGEN	=
LDFLAGS	= 

# ↓ Whether to compile with MPI 
MPI	= false
MPIFC	= mpifort
MPICXX	= mpicxx
MPILC	= $(MPIFC)
# ↑ Set the MPI Fortran, C++ & linking compilers. 

VERSION = $(shell basename $(CURDIR))
EXE	= $(VERSION)

# ↓ Whether to compile with FFTSG/FFTW/clFFT/cuFFT 
FFTSG	= true
FFTW	= true
OPENCL	= true
CUDA	= true

# ↓ Whether to link static libraries of FFTW/clFFT 
FFT_STATIC	= true

# ↓ FFTW libraries directory; not required if installed to the system UNLESS using static libraries
#   For Windows, copy & place the 'libfftw3-3.dll' library file next to the executable 
FFTW_DIR	= PORT_LIBS/linux_lib/fftw-3.3.8-gcc-9.1

# ↓ clFFT libraries directory; not required if installed to the system UNLESS using static libraries
#   For Windows, Copy & place the 'clFFT.dll' library file next to the executable
CLFFT_DIR	= PORT_LIBS/linux_lib/clfft-gcc
OPENCL_LIB	= /usr/local/cuda/lib64
# ↑ Directory of the OpenCL library
#   Liunx 64bit default with NVIDIA: /usr/local/cuda/lib64
#                       with AMD: /opt/amdgpu-pro/lib/x86_64-linux-gnu
#   MinGW default (incl. 'opencl.dll'): /c/Windows/System32/opencl.dll
#   CygWin default (incl. 'opencl.dll'): /cygwin/c/Windows/System32/opencl.dll
#   On MacOS, this is ignored

# ↓ CUDA directory, Linux/MacOS default: /usr/local/cuda
CUDA_DIR	= /usr/local/cuda
NVCC_FLAG	= nvcc -O3 -use_fast_math -std=c++11 -arch=sm_35
#NVCC_FLAG	= nvcc -std=c++11 -g -G #-arch=sm_35 
# ↑ CUDA compiler & flags; modify '-arch' flag according to GPU's compute capability 

# ↓ Whether to use 'sincos' from C libraries to calculate sine & cosine simultaneously.
#   'false' is recommended, unless compiler optimisations are disabled
SINCOS	= false

# ↓	Add FORTRAN code generation flag for ifort/gfortran/pgfortran
ifneq (,$(findstring ifort,$(FC))) 
	FGEN	+= -132 
else ifneq (,$(findstring gfortran,$(FC))) 
	FGEN	+= -ffixed-line-length-132 
else ifneq (,$(findstring flang,$(FC))) 
	FGEN	+= -Mextend 
else ifneq (,$(findstring pgfortran,$(FC))) 
	FGEN	+= -Mextend 
endif
# ↓	Add linker flags for icpc/g++/clang++/pgc++
ifneq (,$(findstring icpc,$(CXX))) 
	LDFLAGS	+= -cxxlib 
else ifneq (,$(findstring g++,$(CXX))) 
	LDFLAGS	+= -lstdc++ 
else ifneq (,$(findstring clang++,$(CXX))) 
	LDFLAGS	+= -lc++ 
else ifneq (,$(findstring pgc++,$(CXX))) 
	LDFLAGS	+= -pgc++libs 
endif

# ↓	Additional flags
FGEN	+= $(OPT) 
CXXGEN	+= $(OPT) -std=c++11 -pthread 
LDFLAGS	+= -pthread -lm

ifeq ($(MPI),true)
	FBFLAGS	+= -D MPI
	FC	= $(MPIFC)
	CXX	= $(MPICXX)
	LC	= $(MPILC)
endif

##########################################################
# ↓	Routines for different platforms/Multi-slice Propagation methods;
#		usually need no modification 

UNAME := $(shell uname -s)

ifeq ($(SINCOS), true)
	FBFLAGS	+= -D SINCOS
	CPPCOMMFLAGS	+= -D SINCOS
endif

ifeq ($(FFTSG), true)
	FFTOBJS	+= fftsg.o fftsg2d.o 
	FBFLAGS	+= -D COMPILE_FFTSG
endif

ifeq ($(FFTW), true)
	FBFLAGS	+= -D COMPILE_FFTW 
	FFTOBJS	+= fftw_func.o 
	ifeq ($(FFT_STATIC), true) 
		LDFLAGS	+= $(FFTW_DIR)/libfftw3.a $(FFTW_DIR)/libfftw3f.a -ldl 
	else 
		ifeq ($(UNAME), Darwin) 
			LDFLAGS	+= -L"$(FFTW_DIR)" -lfftw3 -lfftw3f 
			LDFLAGS += -Wl,-rpath,@executable_path/$(FFTW_DIR)
		else ifeq ($(UNAME), Linux) 
			LDFLAGS	+= -L"$(FFTW_DIR)" -lfftw3 -lfftw3f 
			LDFLAGS	+= -Wl,-rpath='$$ORIGIN'/$(FFTW_DIR)
		else 	# CygWin/MinGW
			LDFLAGS	+= -Wl,-Bdynamic $(FFTW_DIR)/libfftw3-3.dll
		endif
	endif
endif

ifeq ($(OPENCL), true) 
	FFTOBJS	+= opencl_func.o 
	FBFLAGS	+= -D COMPILE_OPENCL 
	ifeq ($(FFT_STATIC), true) 
		LDFLAGS	+= $(CLFFT_DIR)/libclFFT.a -ldl 
		ifeq ($(UNAME), Darwin) 
			LDFLAGS	+= -framework OpenCL 
		else
			LDFLAGS	+= -L"$(OPENCL_LIB)" -lOpenCL
		endif
	else
		ifeq ($(UNAME), Darwin) 
			LDFLAGS	+= -framework OpenCL -L"$(CLFFT_DIR)" -lclFFT 
			LDFLAGS += -Wl,-rpath,@executable_path/$(CLFFT_DIR)
		else ifeq ($(UNAME), Linux)
			LDFLAGS	+= -L"$(OPENCL_LIB)" -lOpenCL -L"$(CLFFT_DIR)" -lclFFT 
			LDFLAGS	+= -Wl,-rpath='$$ORIGIN'/$(CLFFT_DIR)
		else	# CygWin/MinGW
			LDFLAGS	+= -Wl,-Bdynamic $(OPENCL_LIB) $(CLFFT_DIR)/clFFT.dll
		endif
	endif
endif

ifeq ($(CUDA), true) 
	FBFLAGS	+= -D COMPILE_CUDA 
	FFTOBJS	+= cuda_func.a
#	FFTOBJS	+= cuda_func.o
	ifeq ($(UNAME), Darwin)
		CUDA_LIBDIR	= $(CUDA_DIR)/lib
	else ifeq ($(UNAME), Linux)
		CUDA_LIBDIR	= $(CUDA_DIR)/lib64
	endif
	LDFLAGS	+= -L"$(CUDA_LIBDIR)" -lcudart -lcufft 
	LDFLAGS += -Wl,-rpath,$(CUDA_LIBDIR)
endif

# Objects (source files) to build
INCS	= -I"include"
OBJS	= utils.o atomicscatfact.o qcbedms_var.o oneit.o \
		msp_bridge.o cpp_comm.o $(FFTOBJS) \
		parabolic.o amoeba.o anneal.o paranneal.o \
		Multis.o qcbedms_subs.o update.o annepara.o 

#  Build executable
all:    $(EXE)

$(EXE):	$(OBJS)
	$(LC) -o $@ $(OBJS) $(LDFLAGS) 

#  Remove object files, preprocessed so urce files and executable
clean:
	rm  -f $(EXE) $(EXE).exe *.mod *.o *.obj *.lib *.a

################################################
# ↓	Rules for source files

cpp_comm.o: src/cpp_comm.cpp src/cpp_comm.hpp
	$(CXX) -c $< $(CXXGEN) $(CPPCOMMFLAGS) $(INCS) -o $@ 
cuda_func.a: src/cuda_func.cu src/cpp_comm.hpp
	$(NVCC_FLAG) $< -lib -o $@
cuda_func.o: src/cuda_func.cu src/cpp_comm.hpp
	$(NVCC_FLAG) $< -c -o $@
msp_bridge.o: src/msp_bridge.fpp src/qcbedms_var.f
ifneq ($(FFTSG), true)
ifneq ($(FFTW), true)
	$(error At least one of FFTSG and FFTW must be enabled!)
endif
endif
	$(FC) -c $< $(FGEN) $(FBFLAGS) -o $@ 

%.o: src/%.f src/qcbedms_var.f src/utils.f
	$(FC) -c $< $(FGEN) -o $@ 
%.o: src/%.f90 src/qcbedms_var.f src/utils.f
	$(FC) -c $< $(FGEN) -o $@ 
%.o: src/%.cpp src/cpp_comm.hpp
	$(CXX) -c $< $(CXXGEN) $(INCS) -o $@ 
