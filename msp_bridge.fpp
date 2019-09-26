! Multi-slice Propagation (MSP) related Fortran scripts

      module msp_c_wrapper
        implicit none
        interface
! OpenCL & CUDA get total device counts OR print all devices C wrappers
        subroutine OpenCLGetGPU(in,n) bind(c,name="OpenCLGetGPU")
          use iso_c_binding, only: c_int
          integer(c_int),value::in
          integer(c_int)::n
        end subroutine OpenCLGetGPU

        subroutine CUDAGetDevices(in,n) bind(c,name="CUDAGetDevices")
          use iso_c_binding, only: c_int
          integer(c_int),value::in
          integer(c_int)::n
        end subroutine CUDAGetDevices

! MSP libraries global initiation/termination C wrappers
        subroutine fftw_init() bind(c, name="fftw_init")
        end subroutine fftw_init

        subroutine fftw_term() bind(c, name="fftw_term")
        end subroutine fftw_term

        subroutine clfft_init() bind(c, name="clfft_init")
        end subroutine clfft_init

        subroutine clfft_term() bind(c, name="clfft_term")
        end subroutine clfft_term

        subroutine cufft_init() bind(c, name="cufft_init")
        end subroutine cufft_init

        subroutine cufft_term() bind(c, name="cufft_term")
        end subroutine cufft_term

! MSP libraries per child initiation/termination C wrappers
        subroutine fftw_child_init(f) bind(c, name="fftw_child_init")
          use iso_c_binding, only: c_int
          integer(c_int),value::f
        end subroutine fftw_child_init

        subroutine fftw_child_term(f) bind(c, name="fftw_child_term")
          use iso_c_binding, only: c_int
          integer(c_int),value::f
        end subroutine fftw_child_term

        subroutine clfft_child_init(f) bind(c, name="clfft_child_init")
          use iso_c_binding, only: c_int
          integer(c_int),value::f
        end subroutine clfft_child_init

        subroutine clfft_child_term(f) bind(c, name="clfft_child_term")
          use iso_c_binding, only: c_int
          integer(c_int),value::f
        end subroutine clfft_child_term

        subroutine cufft_child_init(f) bind(c, name="cufft_child_init")
          use iso_c_binding, only: c_int
          integer(c_int),value::f
        end subroutine cufft_child_init

        subroutine cufft_child_term(f) bind(c, name="cufft_child_term")
          use iso_c_binding, only: c_int
          integer(c_int),value::f
        end subroutine cufft_child_term

! Hybrid MSP C++11 <thread> wrapper
        subroutine msp_cppthread_exec(nf,offset) bind(c, name='msp_cppthread_exec')
          use iso_c_binding, only: c_int
          integer(c_int),value::nf,offset
        end subroutine msp_cppthread_exec

! One-time FFT & per child MSP C wrappers
        subroutine fftw_one(dir) bind(c, name="fftw_one")
        use iso_c_binding, only: c_int
        integer(c_int),value::dir
        end subroutine fftw_one

        subroutine fftw_msp_f_bridge(f,ct1,ct2) bind(c, name="fftw_msp_f_bridge")
          use iso_c_binding, only: c_int
          integer(c_int),value::f,ct1,ct2
        end subroutine fftw_msp_f_bridge

        subroutine clfft_msp_cast(f,ct1,ct2) bind(c, name="clfft_msp_cast")
          use iso_c_binding, only: c_int
          integer(c_int),value::f,ct1,ct2
        end subroutine clfft_msp_cast

        subroutine cufft_msp_cast(f,ct1,ct2) bind(c, name="cufft_msp_cast")
          use iso_c_binding, only: c_int
          integer(c_int),value::f,ct1,ct2
        end subroutine cufft_msp_cast
        end interface
      end module msp_c_wrapper


      recursive subroutine msp_init_bridge ()
        use qcbedms_var
        use utils
        use iso_c_binding
        use msp_c_wrapper
        implicit none
        logical::fftsg_check,fftw_check,clfft_check,cufft_check
        real(8)::rsum
        integer(c_int)::f
        character(len=8)::list_meth(4),msp_sgl0
        integer::list_n(4),i,j,k

        nhyb=0;  hyb_redi=0;  nfftsg=0;  nfftw=0;  nclfft=0;  ncufft=0
        fftsg_check=.false.;  fftw_check=.false.;  clfft_check=.false.;  cufft_check=.false.
        hybrid_opt=.false.;  msp_sgl=.false.;  MPI_rank=0;  mpi_pernode=.false.

        call read_f7
        read(7,*) msp_sgl0
        call format_case(msp_sgl0,"l")
        if(msp_sgl0(1:1) == 's') then
          msp_sgl=.true.
        end if

        call read_f7
        read(7,*) msp_opt1,msp_opt2,msp_opt3
        call format_case(msp_opt1,"l")

        select case (msp_opt1)
          case ('fftsg','fftw','clfft','cufft')
            nhyb0=1
            allocate(msp_ratio0(nhyb0),msp_meth0(nhyb0),msp_dev0(nhyb0),msp_splt0(nhyb0))

            msp_ratio0(1)=1.d0
            msp_meth0(1)=msp_opt1;  msp_dev0(1)=msp_opt2;  msp_splt0(1)=msp_opt3

            call case_method(1)

          case ('hybrid','mpi')
            hybrid_opt=.true.

            if(msp_opt2 == 0) then
              write(13,*) "Number of MSP methods = 0!"
              stop "Number of hybrid Multi-slice Propagation methods = 0!"
            end if

            if(msp_opt1 == 'mpi' .and. msp_opt2 < 0) mpi_pernode=.true.

            nhyb0=abs(msp_opt2);  hyb_redi=msp_opt3
            allocate(msp_ratio0(nhyb0),msp_meth0(nhyb0),msp_dev0(nhyb0),msp_splt0(nhyb0))

            do f=1,nhyb0
              call read_f7
              read (7,*) msp_ratio0(f),msp_meth0(f),msp_dev0(f),msp_splt0(f)

              call case_method(f)
            end do

          case ('test')
            msp_test=logical(.true.,kind=c_bool)
            if(msp_opt3 < 0) then
              msp_timer=logical(.true.,kind=c_bool)
              msp_opt3=-msp_opt3
            end if

            call test_msp_methods(msp_opt2,msp_opt3)
            return

          case default
            stop 'Unknown Multi-slice Propagation method!'
        end select


! Count total threads required by all MS methods as 'nhyb'
        nhyb=nfftsg+nfftw+nclfft+ncufft
        allocate(msp_ratio(nhyb),msp_meth(nhyb),msp_dev(nhyb),msp_splt(nhyb))

        list_meth=(/ 'fftsg   ','fftw    ','clfft   ','cufft   ' /)
        list_n=(/ nfftsg,nfftw,nclfft,ncufft /)

! Assign each thread/device of the listed MS methods to an individual slot
        j=0
        do f=1,nhyb0
          do k=1,4
            if(msp_meth0(f) == list_meth(k)) then
              if(msp_dev0(f) == -2 .or. msp_meth0(f)(1:3) == 'fft') then
                do i=1,list_n(k)
                  j=j+1
                  msp_ratio(j)=msp_ratio0(f);  msp_meth(j)=msp_meth0(f)
                  msp_dev(j)=i-1;  msp_splt(j)=msp_splt0(f)
                enddo
              else
                j=j+1
                msp_ratio(j)=msp_ratio0(f);  msp_meth(j)=msp_meth0(f)
                msp_dev(j)=msp_dev0(f);  msp_splt(j)=abs(msp_splt0(f))
              endif
            endif
          enddo
        enddo

! Initiate MPI
        if(msp_opt1 == 'mpi') call msp_mpi_init()

! Normalise the ratios of tilt assignment
        rsum=sum(msp_ratio)
        msp_ratio(:)=msp_ratio(:)/rsum

        allocate(as_nt(nhyb),as_toff(nhyb),msp_ntilt(nhyb),msp_spd(nhyb))
        allocate(timer_pr(nhyb),timer_ch(nhyb),timer_fft(nhyb),timer_ifft(nhyb),
     *   timer_mpg(nhyb),timer_mpr(nhyb))

! Assign Fortran array references to shared their C pointers
        cptr_deltaz=c_loc(deltaz);  cptr_xshift=c_loc(xshift);  cptr_yshift=c_loc(yshift)
        cptr_applieddilation=c_loc(applieddilation);  cptr_cell2rpr=c_loc(cell2rpr)
        cptr_beam=c_loc(beam);  cptr_tiltarray=c_loc(tiltarray)
        cptr_res2adr=c_loc(res2adr);  cptr_ib=c_loc(ib);  cptr_slpr=c_loc(slpr)
        cptr_msp_ratio=c_loc(msp_ratio);  cptr_msp_meth=c_loc(msp_meth)
        cptr_msp_dev=c_loc(msp_dev);  cptr_msp_splt=c_loc(msp_splt)
        cptr_as_nt=c_loc(as_nt);  cptr_as_toff=c_loc(as_toff);  cptr_msp_ntilt=c_loc(msp_ntilt)
        cptr_timer_ch=c_loc(timer_ch);  cptr_timer_pr=c_loc(timer_pr)
        cptr_timer_fft=c_loc(timer_fft);  cptr_timer_ifft=c_loc(timer_ifft)
        cptr_timer_mpg=c_loc(timer_mpg);  cptr_timer_mpr=c_loc(timer_mpr)

! Setup each MSP library's global environment
        fft_one_lib=""
#ifdef COMPILE_FFTW
        if(msp_opt1 /= 'fftsg') then
          fft_one_lib='fftw'
        end if
        call fftw_init()
        call c_f_pointer(cptr_pgarray,pgarray,[mt,nslice])
        call c_f_pointer(cptr_pmap0,pmap0,[mt])
#else
        allocate(pgarray(mt,nslice),pmap0(mt))
        cptr_pgarray=c_loc(pgarray);  cptr_pmap0=c_loc(pmap0)
#endif
#ifdef COMPILE_FFTSG
        if(fft_one_lib /= 'fftw') then
          fft_one_lib='fftsg'
          allocate(fsgi(0:2**(int(log(max(meshx,meshy)+0.5d0)/log(2.d0))/2)+1),
     *     fsga(0:2*meshx-1,0:meshy-1),fsgt(0:8*meshy-1),fsgw(0:max(meshx,meshy)/2-1))
          allocate(prarray(mt,npr))
          fsgi(0)=0
        end if
#endif
#ifdef COMPILE_OPENCL
        call clfft_init
#endif
#ifdef COMPILE_CUDA
        call cufft_init
#endif

        write(13,'("Using ",a," for FFT outside multi-slice propagation.")')
     *   fft_one_lib
        if(hybrid_opt) then
          write(13,'(/,"Using ",i0,2x,a," Multi-slice Propagation items (threads).")')
     *     nhyb,trim(msp_opt1)
          if(hyb_redi>0) then
            write(13,'("Auto-redistribute tilts every ",i0," iterations.")') hyb_redi
          end if
        else
          write(13,'(/,"Using ",a," for multi-slice propagation.")') msp_opt1
        end if

! Initiate each MSP method
        call msp_child_init_bridge()

!        call c_shared_var_test
!        call f_c_var_test
      contains
        subroutine case_method(ff)
          implicit none
          integer(c_int)::ff

          select case (msp_meth0(ff))
            case ('fftsg')
              msp_dev0(ff)=1
              nfftsg=msp_dev0(ff)
              if(hybrid_opt) stop 'FFTSG is not implemented in Hybrid Multi-slice!'

            case ('fftw')
              if(msp_dev0(ff) == 0) msp_dev0(ff)=1
              nfftw=msp_dev0(ff)
              if(fftw_check) stop 'Only ONE entity of FFTW is allowed!'
              fftw_check=.true.

            case ('clfft')
              if(msp_dev0(ff) < 0) then
#ifdef COMPILE_OPENCL
                call OpenCLGetGPU(msp_dev0(ff),nclfft)
#endif
                if(clfft_check) stop
     *           'Only ONE entity of clFFT is allowed while device option = -2 (all)!'
                clfft_check=.true.
              else
                nclfft=nclfft+1
              endif

            case ('cufft')
              if(msp_dev0(ff) < 0) then
#ifdef COMPILE_CUDA
                call CUDAGetDevices(msp_dev0(ff),ncufft)
#endif
                if(cufft_check) stop
     *           'Only ONE entity of cuFFT is allowed while device option = -2 (all)!'
                cufft_check=.true.
              else
                ncufft=ncufft+1
              endif

            case default
              print'("Multi-slice method: ",a)', msp_meth0(ff)
              stop "Unknown multi-slice method!"
          end select
        end subroutine case_method
      end subroutine msp_init_bridge


      subroutine msp_term_bridge()
        use qcbedms_var
        use msp_c_wrapper
        implicit none

        call msp_child_term_bridge()

#ifdef COMPILE_FFTW
        call fftw_term
#else
        deallocate(pgarray,pmap0)
#endif
#ifdef COMPILE_OPENCL
        call clfft_term
#endif
#ifdef COMPILE_CUDA
        call cufft_term
#endif

        if(msp_opt1 == 'mpi') call msp_mpi_term()

        if(fft_one_lib == 'fftsg') deallocate(fsga,fsgw,fsgi,fsgt,prarray)
        deallocate(as_nt,as_toff,msp_ntilt,msp_spd)
        deallocate(msp_ratio,msp_meth,msp_dev,msp_splt)
        deallocate(msp_ratio0,msp_meth0,msp_dev0,msp_splt0)
        deallocate(timer_ch,timer_pr,timer_fft,timer_ifft,timer_mpg,timer_mpr)
      end subroutine msp_term_bridge


      subroutine msp_child_init_bridge()
        use qcbedms_var
        use msp_c_wrapper
        implicit none
        integer(c_int)::f,offset,n
! Heterogeneous job & memory allocation, Pt. 1 - Global
! 1. Distribute all tilts to be calculated to each child according to the allocated ratios:
        as_nt(:)=floor(tottilts*msp_ratio(:))
!    If there is any residual tilts caused by flooring, increase tilts from the 1st child:
        do f=1,tottilts-sum(as_nt)
          as_nt(f)=as_nt(f)+1
        end do
!    Note - tottilts === sum( as_nt )
! 2. Range of calculating tilts distributed to Child 'f' -
!     ( 1 + as_toff(f) : as_nt(f) + as_toff(f) ):
        as_toff=0
        do f=2,nhyb
          if(as_nt(f) > 0) as_toff(f)=as_toff(f-1)+as_nt(f-1)
        end do
! 3. Numbers of parallelised tilts for each child will be determined by each's restrictions
!     during their initialisations, satisfying the following rule:
!       msp_ntilt(:) = ceil( as_nt(:) / msp_splt(:) )
!   The memory usage (GPU memory for GPU methods) of each child is dominated by
!       'msp_ntilt(:)' - the number of paralleling tilts of each child.
!   Note - msp_ntilt(:) * msp_splt(:) >= as_nt(:) because of ceiling

        offset=0;  n=nhyb
        if(msp_opt1 == 'mpi') then
          call mpi_calc_tilt_init()
          offset=mhyb_oset(mpir);  n=mhyb(mpir)+mhyb_oset(mpir)
        end if

        do f=1+offset,n
          write(13,'(i0,". ",a," - ",f6.2,"%, ",i0," tilts")')
     *     f,msp_meth(f),msp_ratio(f)*100,as_nt(f)

          select case (msp_meth(f))
          case ("fftsg")
#ifndef COMPILE_FFTSG
            stop "QCBEDMS-PF was compiled without FFTSG!"
#endif
            msp_ntilt(f)=1

          case ("fftw")
#ifdef COMPILE_FFTW
            call fftw_child_init(f-1)
#else
            stop "QCBEDMS-PF was compiled without FFTW!"
#endif

          case ("clfft")
#ifdef COMPILE_OPENCL
            call clfft_child_init(f-1)
#else
            stop "QCBEDMS-PF was compiled without OpenCL!"
#endif

          case ("cufft")
#ifdef COMPILE_CUDA
            call cufft_child_init(f-1)
#else
            stop "QCBEDMS-PF was compiled without CUDA!"
#endif

          case default
            write(*,'("Multi-slice Propagation method: ",a)') msp_meth(f)
            stop "Unknown Multi-slice Propagation method!"
          end select
        end do
      contains
        subroutine mpi_calc_tilt_init()
#ifdef MPI
          use mpi
          integer::nct,ct1,ierr

! Summarise tilts of MPI slave threads
          nct=sum(as_nt(1+mhyb_oset(mpir):mhyb(mpir)+mhyb_oset(mpir)))
          ct1=as_toff(1+mhyb_oset(mpir))

          call MPI_ALLGATHER(nct,1,MPI_INTEGER,mas_nt,1,MPI_INTEGER,MPI_COMM_WORLD,ierr)
          call MPI_ALLGATHER(ct1,1,MPI_INTEGER,mas_toff,1,MPI_INTEGER,MPI_COMM_WORLD,ierr)
!        print*,'mpir',mpir,'mas_toff',mas_toff,'c',as_toff
#endif
        end subroutine mpi_calc_tilt_init

      end subroutine msp_child_init_bridge


      subroutine msp_child_term_bridge()
        use qcbedms_var
        use msp_c_wrapper
        implicit none
        integer::f,offset,n

        offset=0;  n=nhyb
        if(msp_opt1 == 'mpi') then
          offset=mhyb_oset(mpir);  n=mhyb(mpir)+mhyb_oset(mpir)
        end if

        do f=1+offset,n
          select case (msp_meth(f))
            case ("fftsg")
              ! Nothing to do
            case ("fftw")
#ifdef COMPILE_FFTW
              call fftw_child_term(f-1)
#endif
            case ("cufft")
#ifdef COMPILE_CUDA
              call cufft_child_term(f-1)
#endif
            case ("clfft")
#ifdef COMPILE_OPENCL
              call clfft_child_term(f-1)
#endif
          end select
        end do
      end subroutine msp_child_term_bridge


      subroutine msp_redi_calc_ntilt()
        use qcbedms_var
        implicit none
        integer::f
        real(8)::rsum=0

        write(13,'(/,"Redistributing tilts of Multi-slice Propagation methods")')
        msp_ratio(1:nhyb)=msp_spd(:)/hyb_redi
        write(13,'("Measured speeds: ",100(i0,". ",a8," - ",g14.5," tilt/s, "))')
     *   (f,msp_meth(f),msp_ratio(f),f=1,nhyb)

        rsum=sum(msp_ratio(1:nhyb))
        msp_ratio(1:nhyb)=msp_ratio(1:nhyb)/rsum
        write(13,'("Ratio updated: ",100(i0,". ",a8," - ",f7.2,"%, "))')
     *   (f,msp_meth(f),msp_ratio(f)*100,f=1,nhyb)

        call msp_child_term_bridge()
        call msp_child_init_bridge()
        write(13, '("Multi-slice Propagation tilts redistributed!",/)')
      end subroutine msp_redi_calc_ntilt


      subroutine msp_exec_bridge()
        use qcbedms_var
        use msp_c_wrapper
        implicit none

        timer_pr=0;  timer_fft=0;  timer_ifft=0;  timer_mpg=0;  timer_mpr=0

        select case (msp_opt1)
          case ('mpi')
            call msp_mpi_exec
          case default !('hybrid','fftsg','fftw','clfft','cufft')
            call msp_cppthread_exec(nhyb,0)
        end select
      end subroutine msp_exec_bridge


      subroutine msp_mpi_init()
#ifdef MPI
        use qcbedms_var
        use mpi
        implicit none
        integer::ierr,r
        character(len=8,kind=c_char),target,allocatable::imeth(:)
        integer(c_int),target,allocatable::idev(:),isplt(:)
        real(c_double),target,allocatable::iratio(:)
        logical::initd

        call MPI_INITIALIZED(initd,ierr)
        if(.not. initd) call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD,MPI_rank,ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD,nmpi,ierr)

        mpir=MPI_rank+1

        allocate(mhyb(nmpi),mhyb_oset(nmpi),mas_nt(nmpi),mas_toff(nmpi))
        allocate(iratio,source=msp_ratio)
        allocate(imeth,source=msp_meth)
        allocate(idev,source=msp_dev)
        allocate(isplt,source=msp_splt)

! Gather & sync number of MSP methods
        call MPI_ALLGATHER(nhyb,1,MPI_INTEGER,mhyb,1,MPI_INTEGER,MPI_COMM_WORLD,ierr)
        nhyb=sum(mhyb)

! Prepare the new MSP method containers
        deallocate(msp_ratio,msp_meth,msp_dev,msp_splt)
        allocate(msp_ratio(nhyb),msp_meth(nhyb),msp_dev(nhyb),msp_splt(nhyb))

! Prepare displacements
        mhyb_oset(1)=0
        do r=2,nmpi
          mhyb_oset(r)=mhyb_oset(r-1)+mhyb(r-1)
        end do

! Gather & sync parameters of MSP methods
        call MPI_ALLGATHERV(iratio,mhyb(mpir),MPI_DOUBLE_PRECISION,
     *   msp_ratio,mhyb,mhyb_oset,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
        call MPI_ALLGATHERV(imeth,8*mhyb(mpir),MPI_CHARACTER,
     *   msp_meth,8*mhyb,8*mhyb_oset,MPI_CHARACTER,MPI_COMM_WORLD,ierr)
        call MPI_ALLGATHERV(idev,mhyb(mpir),MPI_INTEGER,
     *   msp_dev,mhyb,mhyb_oset,MPI_INTEGER,MPI_COMM_WORLD,ierr)
        call MPI_ALLGATHERV(isplt,mhyb(mpir),MPI_INTEGER,
     *   msp_splt,mhyb,mhyb_oset,MPI_INTEGER,MPI_COMM_WORLD,ierr)

        deallocate(iratio,imeth,idev,isplt)

!        print*,'rank=',MPI_rank,msp_dev
!        print*,'rank=',MPI_rank,msp_meth
#else
        stop 'QCBEDMS-PF was compiled without MPI!'
#endif
      end subroutine msp_mpi_init


      subroutine program_mpi_term()
#ifdef MPI
        use mpi
        integer::ierr
        call MPI_FINALIZE(ierr)
#endif
      end subroutine program_mpi_term


      subroutine msp_mpi_term()
#ifdef MPI
        use qcbedms_var
        use mpi
        if(MPI_rank == 0) call msp_mpi_loops(.true.,p)
        deallocate(mhyb,mhyb_oset,mas_nt,mas_toff)
#endif
      end subroutine msp_mpi_term


      subroutine msp_mpi_loops(opt,pt)
#ifdef MPI
        use qcbedms_var
        use mpi
        integer::ierr,l
        logical::opt
        real(8)::pt(npar)

        l=-1
        if(opt) l=-2
! Broadcase the 'loops' parameter to MPI slave threads
        call MPI_BCAST(l,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
! Broadcase parameters for current iteration
        call MPI_BCAST(PT,npar,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
!        print*,'MPI_rank',MPI_rank,pt

        if(MPI_rank > 0) loops=l
!        print*,MPI_rank,i,loops,ncalcs,initd,finad
#endif
      end subroutine msp_mpi_loops


      subroutine msp_mpi_exec()
#ifdef MPI
        use qcbedms_var
        use mpi
        use msp_c_wrapper
        implicit none
        integer::ierr=0,b

! Execute MSP
        if(mpi_pernode) then
          call msp_cppthread_exec(mhyb(mpir),mhyb_oset(mpir))
        else
          call msp_child_exec_bridge(mpir)
        end if

! Gather calculated tilts from MPI slave threads
        do b=1,mbout
          call MPI_GATHERV(beam(mas_toff(mpir)+1,b),mas_nt(mpir),MPI_DOUBLE_PRECISION,
     *     beam(:,b),mas_nt,mas_toff,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
        end do
!        print'(2i4,1000g15.7)',MPI_rank,ncalcs,beam(:,:)

! Gather Timers
          call MPI_GATHERV(timer_ch(1+mhyb_oset(mpir)),mhyb(mpir),MPI_DOUBLE_PRECISION,
     *     timer_ch,mhyb,mhyb_oset,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
          call MPI_GATHERV(timer_pr(1+mhyb_oset(mpir)),mhyb(mpir),MPI_DOUBLE_PRECISION,
     *     timer_pr,mhyb,mhyb_oset,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
          call MPI_GATHERV(timer_fft(1+mhyb_oset(mpir)),mhyb(mpir),MPI_DOUBLE_PRECISION,
     *     timer_fft,mhyb,mhyb_oset,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
          call MPI_GATHERV(timer_ifft(1+mhyb_oset(mpir)),mhyb(mpir),MPI_DOUBLE_PRECISION,
     *     timer_ifft,mhyb,mhyb_oset,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
          call MPI_GATHERV(timer_mpg(1+mhyb_oset(mpir)),mhyb(mpir),MPI_DOUBLE_PRECISION,
     *     timer_mpg,mhyb,mhyb_oset,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
          call MPI_GATHERV(timer_mpr(1+mhyb_oset(mpir)),mhyb(mpir),MPI_DOUBLE_PRECISION,
     *     timer_mpr,mhyb,mhyb_oset,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
#endif
      end subroutine msp_mpi_exec


      recursive subroutine msp_child_exec_bridge(f)
        use iso_c_binding,only:c_int
        interface
          recursive subroutine msp_child_exec_bridge_f(f)
     *     bind(c,name='msp_child_exec_bridge_f')
            use iso_c_binding,only:c_int
            integer(c_int),value::f
          end subroutine msp_child_exec_bridge_f
        end interface

        integer(c_int)::f
        call msp_child_exec_bridge_f(f)
      end subroutine msp_child_exec_bridge


      recursive subroutine msp_child_exec_bridge_f(f) bind(c,name='msp_child_exec_bridge_f')
        use utils
        use qcbedms_var
        use msp_c_wrapper
        implicit none
        integer(c_int),value::f
        integer(c_int)::ct1,ct2
        integer(8)::t_ch

        call system_clock(t_ch)

        ct1=as_toff(f)
        ct2=as_toff(f)+as_nt(f)

        select case (msp_meth(f))
          case ("fftsg")
#ifdef COMPILE_FFTSG
            call fftsg_msp(f,pgarray,prarray,pmap0)
#endif

          case ("fftw")
#ifdef COMPILE_FFTW
            call fftw_msp_f_bridge(f-1,ct1,ct2)
#endif

          case ("clfft")
#ifdef COMPILE_OPENCL
            call clfft_msp_cast(f-1,ct1,ct2)
#endif

          case ("cufft")
#ifdef COMPILE_CUDA
            call cufft_msp_cast(f-1,ct1,ct2)
#endif
        end select
!        write(*,'(1000g25.16)') beam(:,1)
        timer_ch(f)=timer(t_ch)
      end subroutine msp_child_exec_bridge_f


      subroutine fft_one(dir,lib)
        use qcbedms_var
        use msp_c_wrapper
        implicit none
        integer,intent(in)::dir
        character(len=*,kind=c_char),intent(in)::lib
        integer::i,j,j1

        if(lib == "fftw") then
#ifdef COMPILE_FFTW
          call fftw_one(int(dir,c_int))
#endif
        else
#ifdef COMPILE_FFTSG
C
C FAST FOURIER TRANSFORM USING 'cdft2d' FROM 'fftsg2d',
C BY Takuya OOURA, Research Institute for Mathematical Sciences,
C Kyoto University, Kyoto 606-01 Japan
C http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
C
          DO J=0,MESHY-1
            j1=J*MESHX
            DO I=0,MESHX-1
              fsga(2*I,J)=DBLE(pmap0(I+1+j1))
              fsga(2*I+1,J)=AIMAG(pmap0(I+1+j1))
            END DO
          END DO

          CALL CDFT2D(2*MESHX,2*MESHX,MESHY,dir,fsga,fsgt,fsgi,fsgw)

          DO J=0,MESHY-1
            j1=J*MESHX
            DO I=0,MESHX-1
              PMAP0(I+1+j1)=CMPLX(fsga(2*I,J),fsga(2*I+1,J),kind=8)
            END DO
          END DO
#endif
        end if
      end subroutine fft_one


#ifdef COMPILE_FFTSG
      subroutine fftsg_msp(f,pg,pr,pmap)
! Start MSP using FFTSG
        use utils
        use qcbedms_var
        implicit none
        integer(c_int)::f
        integer::n,t,npg,res2t
        integer(8)::t_msp
        complex(8)::pg(mt,nslice),pr(mt,npr),pmap(mt)
        real(8)::dzdil(npr)

        dzdil=deltaz*APPLIEDDILATION

        do t=1,as_nt(f)
          res2t=res2adr(t+as_toff(f))

          DO n=1,npr
            CALL PROPG(TILTARRAY(:,t,n),cell2rpr(:,n),dzdil(n),XSHIFT(n),YSHIFT(n),pr(:,n))
          END DO

C MULTISLICE LOOP
          pmap(1:mt)=cmplx(1,0,kind=8)

          if(msp_timer) call system_clock(t_msp)
          DO n=1,sl_hi
            NPG=MSEQ(n)
C            CALL UTEST(N,pmap0)
C            pmap=pmap/TOTALINT

            pmap(1:mt)=pmap(1:mt)*pg(1:mt,npg)
            if(msp_timer) timer_mpg(f)=timer_mpg(f)+timer(t_msp)

            call fft_one(-1,"fftsg")
            if(msp_timer) timer_fft(f)=timer_fft(f)+timer(t_msp)

            pmap(1:mt)=pmap(1:mt)*pr(1:mt,slpr(npg))
            if(msp_timer) timer_mpr(f)=timer_mpr(f)+timer(t_msp)

            if((sl_opt .and. n >= sl_lo) .or. n == sl_hi) then
              if(sl_opt) then
                sl_beam(t,1:mbout,n-sl_lo+1)=abs(pmap(IB(1:mbout)))**2
              else
                BEAM(t,1:mbout)=abs(pmap(IB(1:mbout)))**2
              end if

              if(n == sl_hi) exit
            end if

            if(msp_timer) call system_clock(t_msp)
            call fft_one(1,"fftsg")
            if(msp_timer) timer_ifft(f)=timer_ifft(f)+timer(t_msp)
          END DO
C         CALL UTEST(N,pmap0)
        END DO

! End MSP using FFTSG
      end subroutine fftsg_msp
#endif


      recursive subroutine fftw_msp_f(f,ct1,ct2,pg,pr,pmap) bind(c,name='fftw_msp_f')
#ifdef COMPILE_FFTW
        use utils
        use qcbedms_var
        implicit none
        interface
          subroutine fftw_fft(f,dir) bind(c,name='fftw_fft')
            use iso_c_binding, only: c_int
            integer(c_int),value::f,dir
          end subroutine fftw_fft

          subroutine fftw_time_now(f) bind(c,name='fftw_time_now')
            use iso_c_binding, only: c_int
            integer(c_int),value::f
          end subroutine fftw_time_now

          function fftw_timer(f) bind(c,name='fftw_timer')
            use iso_c_binding, only: c_int,c_double
            integer(c_int),value::f
            real(c_double)::fftw_timer
          end function fftw_timer
        end interface

        integer(c_int),value::f,ct1,ct2
        complex(c_double_complex)::pg(mt,nslice),pr(mt,npr),pmap(mt)
        integer::k,t,kpg
        real(8)::dzdil(npr)

        dzdil=deltaz*APPLIEDDILATION

        do t=ct1,min(ct2,as_nt(f)+as_toff(f))
! Prepare the Fresnel propagators 'pr'
          if(msp_timer) call fftw_time_now(f-1)
          DO k=1,npr
            CALL PROPG(TILTARRAY(:,t,k),cell2rpr(:,k),dzdil(k),XSHIFT(k),YSHIFT(k),pr(:,k))
!      if(t==1) write(*,'(a,/,4(2(g0.7,x),2x))') 'pr',pr(:,k)
          END DO
          if(msp_timer) timer_pr(f)=timer_pr(f)+fftw_timer(f-1)

          pmap(:)=cmplx(1,0,kind=8)

! Multi-slice propagation
          if(msp_timer) call fftw_time_now(f-1)
          DO k=1,sl_hi
            kpg=MSEQ(k)
! nature-order pg
            pmap(:)=pmap(:)*pg(:,kpg)
            if(msp_timer) timer_mpg(f)=timer_mpg(f)+fftw_timer(f-1)

! nature-order fft input
            call fftw_fft(f-1,-1)
! O-shifted-order wave: 000 at pmap(0)
            if(msp_timer) timer_fft(f)=timer_fft(f)+fftw_timer(f-1)

! O-shifted pr
            pmap(:)=pmap(:)*pr(:,slpr(kpg))
            if(msp_timer) timer_mpr(f)=timer_mpr(f)+fftw_timer(f-1)

            if((sl_opt .and. k >= sl_lo) .or. k == sl_hi) then
! O-shifted 'pmap' expected by 'ib'
              if(sl_opt) then
                sl_beam(t,1:mbout,k-sl_lo+1)=abs(pmap(ib(1:mbout)))**2
              else
                beam(t,1:mbout)=abs(pmap(ib(1:mbout)))**2
              end if

              if(k == sl_hi) exit
            end if

! O-shifted ifft input
            call fftw_fft(f-1,1)
! nature-order wave: 000 (kind of) at centre
            if(msp_timer) timer_ifft(f)=timer_ifft(f)+fftw_timer(f-1)
          end do
!      if(t==1) write(*,'(a,/,4(2(g0.7,x),2x))') 'exit wave',pmap(1:mt)
        end do
#endif
      end subroutine fftw_msp_f


      recursive subroutine sincos(x,sn,cs)
        implicit none
        real(8)::x,sn,cs
#ifdef SINCOS
        interface
          subroutine c_sincos(x,sn,cs) bind(c,name="c_sincos")
            use iso_c_binding, only: c_double
            real(c_double)::x,sn,cs
          end subroutine c_sincos
        end interface
        call c_sincos(x,sn,cs)
#else
        sn=sin(x);  cs=cos(x)
#endif
      end subroutine sincos


      subroutine test_msp_methods(ntest0,nm0)
        use iteration
        IMPLICIT none
        integer::ntest,nm,ntest0,nm0,i,j,f,jmin(1)
        real(8)::chii
        real(8),allocatable::t1i(:),tch(:,:),tpr(:,:),
     *   tff(:,:),tfb(:,:),tmg(:,:),tmr(:,:),chi(:),fr(:,:)
        integer,allocatable::nct(:,:)
        character(len=10)::l

        nm=nm0;  ntest=ntest0
        if(nm < 1) nm=1

        open(21, file="test_info.txt", status="unknown", position="append", action="write")

        write (21,'(/,"#-Loop     MSP     Dev Spl    %     Tilts    Tilt/s        1-iter.(s)     ",
     *   "PGT(s)       MS-FFT(s)     MS-IFFT(s)    MS-M⋅MG(s)    MS-M⋅MR(s)         χ         ")')

        do i=1,ntest
          ncalcs=0
          call msp_init_bridge

          allocate(chi(nm),fr(nhyb,nm),nct(nhyb,nm),t1i(nm),tch(nhyb,nm),tpr(nhyb,nm),
     *     tff(nhyb,nm),tfb(nhyb,nm),tmg(nhyb,nm),tmr(nhyb,nm))

          do j=1,nm
            chii=ONEIT(P)
            if(MPI_rank == 0) then
              print '(i0,".",i0,":",g14.5," s, χ =",g16.7)',i,j,timer_1it,chii

              chi(j)=chii
              t1i(j)=timer_1it;   tch(:,j)=timer_ch;    tpr(:,j)=timer_pr
              tff(:,j)=timer_fft; tfb(:,j)=timer_ifft;  tmg(:,j)=timer_mpg
              tmr(:,j)=timer_mpr; fr(:,j)=msp_ratio;    nct(:,j)=as_nt

              write(l,'(i0,"-",i0)') i,j
              call print_data(l)
            end if
          end do

          if(nm>1) then
            chii=sum(chi(:))/nm;  timer_1it=sum(t1i(:))/nm
            do f=1,nhyb
              timer_ch(f)=sum(tch(f,:))/nm;  timer_pr(f)=sum(tpr(f,:))/nm
              timer_fft(f)=sum(tff(f,:))/nm;  timer_ifft(f)=sum(tfb(f,:))/nm
              timer_mpg(f)=sum(tmg(f,:))/nm;  timer_mpr(f)=sum(tmr(f,:))/nm
              msp_ratio(f)=sum(fr(f,:))/nm;  as_nt(f)=sum(nct(f,:))/nm
            end do
            write(l,'("Avg-",i0)') i
            call print_data(l)

            jmin=minloc(t1i(:))
            j=jmin(1)
            chii=chi(j)
            timer_1it=t1i(j);   timer_ch=tch(:,j);    timer_pr=tpr(:,j)
            timer_fft=tff(:,j); timer_ifft=tfb(:,j);  timer_mpg=tmg(:,j)
            timer_mpr=tmr(:,j); msp_ratio=fr(:,j);    as_nt=nct(:,j)
            write(l,'("F",i0,"-",i0)') i,j
            call print_data(l)
          end if

          deallocate(chi,fr,nct,t1i,tch,tpr,tff,tfb,tmg,tmr)
          call msp_term_bridge
        end do

        if(MPI_rank == 0) then
          write (21,'("  Reflection mesh = ",i0," * ",i0,", calculated tilts = ",i0)',
     *     advance='no') meshx,meshy,tottilts
          if(sl_opt) then
            write (21,'(", optimum slices search between ",i0," - ",i0)') sl_lo,sl_hi
          else
            write (21,'(", slices = ",i0)') sl_hi
          end if
          close(21)
          write(13,'(/,a)') "End of test!"
          print*,'Test finished!'
        end if

      contains
        subroutine print_data(l)
          use qcbedms_var
          IMPLICIT none
          character(len=10)::l
          if(hybrid_opt) then
            write (21,'(a,t11,"Hybrid  ",2i4,2("    -   "),2g14.5,5(8x,"-",5x),g16.7)')
     *       trim(l),nhyb,msp_opt3,tottilts/timer_1it,timer_1it,chii
            do f=1,nhyb
              write (21,'(a,".",i0,t13,a6,2i4,f8.3,i8,7g14.5,8x,"-")')
     *         trim(l),f,msp_meth(f),msp_dev(f),msp_splt(f),
     *         msp_ratio(f)*100,as_nt(f),as_nt(f)/timer_ch(f),timer_ch(f),timer_pr(f),
     *         timer_fft(f),timer_ifft(f),timer_mpg(f),timer_mpr(f)
            end do
          else
            write (21,'(a,t11,a8,2i4,2("    -   "),7g14.5,g16.7)')
     *       trim(l),msp_meth0(1),msp_dev0(1),msp_splt0(1),tottilts/timer_1it,
     *       timer_1it,maxval(timer_pr),maxval(timer_fft),maxval(timer_ifft),
     *       maxval(timer_mpg),maxval(timer_mpr),chii
          end if
        end subroutine print_data
      end subroutine test_msp_methods


