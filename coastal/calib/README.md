# Overview
This directory contains shell and slurm scripts to make SCHISM calibration runs using NWM retrospective streamflow and forcing, STOFS forecast or TPXO water level forecast as inputs. 

# Prerequisite
1) The OTPSnc program and the TPXO10_atlas data. The OTPSnc program can be downloaded from  https://www.tpxo.net/otps. The TPXO10_atlas data is available at the AWS s3 bucket s3://ngwpc-data/Coastal_and_atmospheric_forcing_for_calibration/TPXO_atlas/TPXO10_atlas_v2_nc.zip. To build the OTPSnc program on the Parallel Works cluster, first untar the tar file, OTPSnc.tar.Z, then update the makefile in the OTPS folder as the following,
----------------------------------------------------------
ARCH = $(shell uname -s)
ifeq ($(ARCH),Linux)
 FC = /contrib/software/gcc/8.5.0/bin/gfortran
 NCLIB = /contrib/software/netcdf/4.7.4/lib
 NCINCLUDE = /contrib/software/netcdf/4.7.4/include
 NCLIBS= -lnetcdf -lnetcdff
endif

predict_tide: predict_tide.f90 subs.f90 constit.h
        $(FC) -o predict_tide predict_tide.f90 subs.f90 -L$(NCLIB) $(NCLIBS) -I$(NCINCLUDE) 
        #rm *.o
extract_HC:  extract_HC.f90 subs.f90
        $(FC) -o  extract_HC extract_HC.f90 subs.f90 -L$(NCLIB) $(NCLIBS) -I$(NCINCLUDE)
        #rm *.o
----------------------------------------------------------
Then run the command 'make'. A binary file called 'predict_tide' will be created.

The TPXO10_atlas data files (TPXO10_atlas_v2_nc.zip) must be unpacked and added to the OTPSnc folder.

There is already an installation of OTPSnc and TPXO10_atlas on the Parallel Works system. It is located at /contrib/software/OTPSnc.

2) An installation of NWM v3. The NWM v3 package is located at the Parallel Works AWS S3 bucket named 'TESTBUCKET', that is s3://6668bc97ecf9731800ee83e8. See the instructions in the file s3://6668bc97ecf9731800ee83e8/nwmv3_oe_install_rocky8/README.TXT about the installation procedures.

However, note that the SCHISM binary from the S3 bucket was compiled for the AWS' hpc7a node. If your cluster has a different node type, the SCHISM binary must be rebuilt. Below is a build script for the Parallel Works NextGen OE cluster.

----------------------------------------------------------
#!/usr/bin/bash

SORCnwm=`pwd`
EXECnwm=`pwd`/../../exec
#MODULEFILESnwm=`pwd`/../../modulefiles
#export MODULEPATH=$MODULEPATH:$MODULEFILESnwm
#
#source ../../versions/build.ver
#module purge
#module load nwm/${nwmVer}
#module list

export PATH=/contrib/software/intel_19_1_3_304_opt/impi/2019.9.304/intel64/bin:/contrib/software/intel_19_1_3_304_opt/compilers_and_libraries/linux/mpi/intel64/bin:/contrib/software/intel_19_1_3_304_opt/bin:$PATH
export LD_LIBRARY_PATH=/contrib/software/intel_19_1_3_304_opt/impi/2019.9.304/intel64/lib:/contrib/software/intel_19_1_3_304_opt/compilers_and_libraries/linux/mpi/intel64/lib:/contrib/software/intel_19_1_3_304_opt/lib:/contrib/software/intel_19_1_3_304_opt/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH

export CC=icc
export CXX=icpc
export FC=ifort
export F77=ifort

export LDFLAGS="-L/contrib/Zhengtao.Cui/home/ngwpc/contrib/software/intel_19_1_3_304_opt/hdf5_1.12.3/lib" 

export NETCDF=/contrib/Zhengtao.Cui/home/ngwpc/contrib/software/intel_19_1_3_304_opt/netcdf

export  NetCDF_INCLUDE_DIR=$NETCDF/include
export  NetCDF_LIBRARIES=$NETCDF/lib
export  NetCDF_FORTRAN_DIR=$NETCDF

export HDF5=/contrib/Zhengtao.Cui/home/ngwpc/contrib/software/intel_19_1_3_304_opt/hdf5_1.12.3
export LD_LIBRARY_PATH=${NETCDF}/lib:${HDF5}/lib:$LD_LIBRARY_PATH

#alias python='python3'
mkdir ./build

cd ./build; rm -rf *

cmake -C ../SCHISM.local.build -C ../SCHISM.local.aws_intel19 ../src/
make -j8 pschism
cd ../

cp ./build/bin/pschism_wcoss2_NO_PARMETIS_TVD-VL ../../exec/pschism_wcoss2_NO_PARMETIS_TVD-VL_intel
----------------------------------------------------------

Run this script to re-build the SCHISM binary on the NextGen OEs.

There is already installed NWM v3 on the Parallel Works systems. It is located at /contrib/software/nwmv3_oe_install

3) Historical STOFS archives downloaded from https://noaa-gestofs-pds.s3.amazonaws.com/index.html

4) Retrospective NWM streamflow and forcing downloaded from https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/index.html 

5) Hot restart file if a hot restart is needed for the calibration

# Script description (without Singularity containers)

run_coastal_workflow.bash : The main driver of the calibration workflow. Part of the environmental variables must be adjusted by the user to fit his/her specific use case.
regrid_stofs.bash  : Regrid the STOFS forecast file into the SCHISM elev2D.th.nc file when STOFS data is used for the boundary conditions. 
make_tpxo_ocean.bash : Create SCHISM boundary condition file from the TPXO10 Atlas dataset.
nwm_forcing_coastal.bash : Create atmospheric input files for SCHISM from the NWM forcing.
initial_discharge.bash   : Create the discharge file from NWM forcing.
combine_sink_source.bash : Combine the sink and source files
merge_source_sink.bash   : Merge the sink and source files
update_param.bash        : Create the SCHISM parm.nml namelist file according to user's settings.
nwm_coastal.bash         : Run the SCHISM using the input files created by the scripts above.

# Script description (with Singularity containers)
sing_run.bash : The main driver of the calibration workflow when singurlarity containter is used. Part of the environmental variables must be adjusted by the user to fit his/her specific use case.
run_sing_coastal_workflow_pre_forcing_coastal.bash : This script is executed by the Singularity container to prepare for the execution of workflow_driver.py that requires mpy4py.
pre_nwm_forcing_coastal.bash : Evoked by run_sing_coastal_workflow_pre_forcing_coastal.bash. Prepare data for the execution of workflow_driver.py.
run_sing_coastal_workflow_post_forcing_coastal.bash : This script is executed by the Singularity container for clean up the execution of workflow_driver.py.
post_nwm_forcing_coastal.bash : Evoked by run_sing_coastal_workflow_post_forcing_coastal.bash. Performs the tasks needed after the execution of workflow_driver.py.
run_sing_coastal_workflow_update_params.bash : This script is executed by the Singularity container to create the parameter file for SCHISM.
run_sing_coastal_workflow_make_tpxo_ocean.bash : This script is executed by the Singularity container to create the water level boundary forcing file for SCHISM.
make_tpxo_ocean.bash : This script is evoked by run_sing_coastal_workflow_make_tpxo_ocean.bash to create water level boundary forcing files using the TPXO program and dataset.
run_sing_coastal_workflow_pre_make_stofs_ocean.bash : This script executed by the Singularity container to prepare the execution of regrid_estofs.py.
pre_regrid_stofs.bash : This script is evoked by run_sing_coastal_workflow_pre_make_stofs_ocean.bash to prepare data for the execution of regrid_estofs.py.
run_sing_coastal_workflow_post_make_stofs_ocean.bash : This script is executed by the Singularity container to post-process data created by regrid_estofs.py which requires MPI.
post_regrid_stofs.bash : This script is evoked by run_sing_coastal_workflow_post_make_stofs_ocean.bash to post-process data created by regrid_estofs.py.
run_sing_coastal_workflow_pre_schism.bash : This script is executed by the Singularity container to prepare the execution of SCHISM which requires MPI.
pre_schism.bash : This script is evoked by run_sing_coastal_workflow_pre_schism.bash to prepare the execution of SCHISM.
run_sing_coastal_workflow_post_schism.bash : This script is executed by the Singularity container to post-process the execution of SCHISM.
post_schism.bash : This script is evoked by run_sing_coastal_workflow_post_schism.bash to post-process SCHISM output files.


# Script Usage (without Singularity container)

Update the environmental variables between lines 1 and 66 in the main driver script, run_coastal_workflow.bash.
Lines 1 to 14 are the Slurm job setting.

Lines 15 to 66 are the environmental variable setting for the calibration run. See the comments for the explanation of each variable.

Note that the Slurm job must be submitted from the ngen-forcing/coastal/calib folder.

# Script Usage (with Singularity container)
Update the SLURM settings and environmental variables between lines 1 and 50 in the main driver script for Singularity containers, sing_run.bash.
Lines 3 to 7 are the Slurm job setting.
Lines 8 to 50 are the environmental variable setting for the calibration run. See the comments for the explanation of each variable.
Not that the the variables NODES and NCORES must match with the SLURM job settings defined in lines 3 to 7.

#### Examples ####

Without a Singularity container:

cd ngen-forcing/coastal/calib
sbatch run_coastal_workflow.bash

To check the job status, run the 'squeue' command. To cancel the job, issue the 'scancel <jobid>' command.

With a Singularity container:
sbatch ngen-forcing/coastal/calib/sing_run.bash

# Download scripts:
download_nwm_ana_archived_chout.bash : download the archived NWM AnA cycle streamfloww data for a given time period and domain. It takes three arguments, start date, end date and a domain name. The dates are in the formation of 'YYYYMMDD' and the domain name is one of hawaii, puertorico, atlgulf and pacific.
#### Examples ####
./download_nwm_ana_archived_chout.bash 20240401 20240405 hawaii 
The downloaded files will be saved in the current directory.

download_nwm_ana_archived_forcing.bash : download the archived NWM AnA cycle forcing data for a given time period and domain. It takes three arguments, start date, end date and a domain name. The dates are in the formation of 'YYYYMMDD' and the domain name is one of hawaii, puertorico, atlgulf and pacific.
#### Examples ####
./download_nwm_ana_archived_forcing.bash 20240401 20240405 hawaii 
The downloaded files will be saved in the current directory.
download_nwm_ana_archived_forcing.bash

download_stofs.bash : download the archived STOFS data for a given time period. It takes two arguments, start date and end date. The dates are in the formation of 'YYYYMMDD'.
#### Examples ####
./download_stofs.bash 20240401 20240405
The downloaded files will be saved in the subdirectory of current directory with an pattern of 'stofs_YYYYMMDD' as the subdirectory name.
