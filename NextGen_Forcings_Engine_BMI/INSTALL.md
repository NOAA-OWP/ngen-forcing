# NextGen Forcings Engine Dependencies
•	Please see detailed description of Python environmental dependencies and packages within the pyproject.toml file in the root of this directory, or as specified within the requirements.txt or environment.yml file.

•	Install wgrib2 NCAR tool (Download and compile wgrib2 executable from NCAR (https://github.com/NOAA-EMC/wgrib2). Once you compile the wgrib2 executable and shared libraries, then you can either link the Linux variable “WGRIB2” to the wgrib2 executable (export WGRIB2=/pathway/to/wgrib2_executable) OR install the pywgrib2 Python module as described here (https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/pywgrib2_s_install.html). Either method will work for the NextGen Forcings to utilize the wgrib2 tool internally to convert wgrib2 files into netcdf files.   

# Steps for directly installing the NextGen Forcings Engine onto your own local Python environment using Anaconda (no supercomputer implementation)
2. Execute the following command to create a new Python environment containing all required NextGen Forcings Engine dependencies listed within the .yml file, assuming you're already in the same directory as the environment.yml file located in this directory: conda env create --name NextGen_Forcings_Engine --file=environments.yml

# Steps for Installing the NextGen Forcings Engine with ESMF Library Dependencies
1.	You will first need to load the Intel and netcdf4 libraries precompiled on a given cluster and manually install the ESMF libraries if the forcing engine will be implemented on a given supercomputer. Otherwise, allow the anaconda environment installer to build/link intel libraries to your Python environment for non-supercomputer clusters (as shown in the previous bulletpoint). The following options below are examples for loading up the intel compilers on a given NOAA RDHPCS cluster:
• module load intel/2022.1.2
• module load impi/2022.1.2
• module load netcdf-hdf5parallel/4.7.4
• export FC=mpiifort, export CXX=mpiicpc, export CC=mpiicc

2.	Download and install and ESMF version release >= 8.1.0 from the GitHub repository (https://github.com/esmf-org/esmf). Instructions below highlight method to manually install and link ESMF libraries to intel libraries on a given supercomputer cluster.
  • unzip esmf zipped file downloaded from the GitHub repository, cd into esmf-release-VERSION directory.
  • export ESMF_DIR=/pathway/to/esmf-release-VERSION
  • export ESMF_COMPILER=intel
  • export ESMF_COMM=intelmpi
  • export ESMF_OPENMP=ON
  • export netcdf variables to force ESMF to build with netcdf capabilities (export ESMF_NETCDF=”split”, export ESMF_NETCDF_INCLUDE=$NETCDF_INCLUDE_PATHWAY, export ESMF_NETCDF_LIBPATH=$NETCDF_LIB_PATHWAY,export ESMF_NETCDF_LIBS="-lnetcdff -lnetcdf")
  • gmake
  • gmake install
  • gmake installcheck

3.	*** Need wgrib2 tool as well **** Download and  wgrib2 executable from GitHub (https://github.com/NOAA-EMC/wgrib2),
  • Unzip the wgrib2 tar ball, go into directory and edit "CMakeLists.txt" file and include USE_NETCDF4=ON, BUILD_SHARED_LIB=ON options
  • export CC=gcc; FC=gfortran
  • follow cmake build instructions as stated in the GitHub repository README.md file
  • export WGRIB2=/pathway/to/executable (or compile your anaconda environment with pywgrib2_s python module using instructions here https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/pywgrib2_s_install.html)

4. Install a new Python environment to link the intel (if you've just compiled wgrib2 executable, reload your compiler pathways as shown here export FC=mpiifort, export CXX=mpiicpc, export CC=mpiicc):
  • conda create -n ngen_engine -c conda-forge python numpy pandas bmipy pyyaml
  • conda activate ngen_engine (this will load the python and pip executables, which will be used to install Hera precompiled libraries).
5.	Install mpi4py python libraries using pip install that will configure the mpi4py library to the Hera intel compilers:
  • pip install --no-cache-dir mpi4py”
6.	Install python “setuptools” to properly configure the correct python setup tools needed to setup and install ESMF packages for the anaconda environment.This step is only necessary if you are manually installing/linking ESMF libraries to your Python environment on a supercomputer:
  • pip install setuptools==58.2.0 (Only if you're using the earlier versions of ESMF (8.1.0), otherwise not likely needed to build the ESMF Python module
7.	 Install netcdf4 libraries to be able to load NWM domain files and meteorological datasets to execute the NWMv3.0 Forcings Engine: 
  • pip install --no-cache-dir netCDF4
8.	Install scipy libraries to load up interpolation methods that are utilized within the NextGen Forcings Engine:
  • pip install scipy
9.	cd to esmf-release-VERSION/src/addon/ESMPy directory. This is where the setup.py python script is located to install ESMF libraries onto your anaconda environment. Execute the following steps to link the ESMPy Python module to your anaconda environment:
  • Export ESMFMKFILE=/pathway/to/directory/esmf-release-8.1.0/lib/libO/Linux.intel.64.intel.default/esmf.mk
  • pip install . (earlier versions of ESMF (8.1) have different instructions on compiling the ESMF Python module and should follow accordingly based on their own Python documentation steps)
10.	ESMF packages should be properly installed and linked to the intel compiler libraries for MPI capabilities. Open a python command line and “import ESMF” to make sure that the ESMF libraries were properly install on the anaconda environment. If it succeeds, then you should be good to go with the ESMF libraries!
11. pip install -r requirements.txt - This will allow you to install the remaining Python dependencies required to execute the NextGen Forcings Engine BMI. 
