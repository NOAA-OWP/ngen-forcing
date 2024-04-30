# NextGen Forcings Engine Dependencies
•	Please see detailed description of Python environmental dependencies and packages within the pyproject.toml file in the root of this directory.

•	wgrib2 NCAR tool installation (Download and compile wgrib2 executable from NCAR (https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/wgrib2_v3.1.1_changes.html, execute the precompiled MakeFile in the directory and allow the wgrib2 tool to compiled its own pre-defined libraries and create its wgrib2 executable. Export the Linux variable “WGRIB2” to the for the NWMv3.0 python executable to reference to for converting wgrib2 files to netcdf files (export WGRIB2=/pathway/to/executable).

•	NOAA RDHPCS Hera cluster recommended Rocky8 modules required to link the NextGen Forcings Engine python environment and its MPI communicator to cluster compute nodes/cpus:  
  1. intel/2022.1.2
  2. impi/2022.1.2
  3. netcdf-hdf5parallel/4.7.4
  4. Any ESMF version >= 8.1.0 (contains ESMPy python bindings to wrap with the module installation). Ensure to manually install your own intel dependencies within your anaconda environment that will be properly linked to your ESMF libraries.

# Steps for directly installing the NextGen Forcings Engine onto your own local Python environment using Anaconda (no supercomputer implementation)
1. Make sure cd into the NextGen Forcings Engine BMI directory of this repository.
2. Execute the following command to create a new Python environment containing all required NextGen Forcings Engine dependencies listed within the .yml file: conda env create --name NextGen_Forcings_Engine --file=environments.yml

# Steps for Installing the NextGen Forcings Engine on the RDHPCS Hera Cluster with ESMF Library Dependencies
1.	You will first need to load the ESMF libraries precompiled on the Hera cluster, followed by its respective intel MPI compilers that were used to compile the ESMF code on the supercomputer. Otherwise, allow the anaconda environment installer to build/link intel libraries to your Python environment for non-supercomputer clusters. The following options below are loading the correct compilers up on the RDHPCS Hera Cluster:
• module load intel/2022.1.2
• module load impi/2022.1.2
• module load netcdf-hdf5parallel/4.7.4
• export FC=mpiifort, export CXX=mpiicpc, export CC=mpiicc

3.	Download and install and ESMF version release >= 8.1.0 from the GitHub repository (https://github.com/esmf-org/esmf). Instructions below highlight method to manually install and link ESMF libraries to intel libraries on a given supercomputer cluster.
  • unzip esmf zipped file downloaded from the GitHub repository, cd into esmf-release-VERSION directory.
  • export ESMF_DIR=/pathway/to/esmf-release-VERSION
  • export ESMF_COMPILER=intel
  • export ESMF_COMM=intelmpi
  • export ESMF_OPENMP=ON
  • export netcdf variables to force ESMF to build with netcdf capabilities (export ESMF_NETCDF=”split”, export ESMF_NETCDF_INCLUDE=$NETCDF_INCLUDE_PATHWAY, export ESMF_NETCDF_LIBPATH=$NETCDF_LIB_PATHWAY,export ESMF_NETCDF_LIBS="-lnetcdff -lnetcdf")
  • gmake
  • gmake install
  • gmake installcheck

4.	*** Need wgrib2 tool as well **** Download and  wgrib2 executable from GitHub (https://github.com/weathersource/wgrib2),
  • Unzip the wgrib2 tar ball, go into directory and edit "makefile" to include USE_NETCDF4=1, MAKE_SHARED_LIB=1 options
  • export CC=gcc; FC=gfortran
  • make
  • make lib (will include shared library libwgrib2.so in case user wants to link library with pywgrib2_s module)
  • export WGRIB2=/pathway/to/executable (or compile your anaconda environment with pywgrib2_s python module using instructions here https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/pywgrib2_s_install.html)

7. Install a new Python environment to link the intel (if you've just compiled wgrib2 executable, reload your compiler pathways as shown here export FC=mpiifort, export CXX=mpiicpc, export CC=mpiicc):
  • conda create -n ngen_engine -c conda-forge python numpy pandas
  • conda activate ngen_engine (this will load the python and pip executables, which will be used to install Hera precompiled libraries).
6.	Install mpi4py python libraries using pip install that will configure the mpi4py library to the Hera intel compilers:
  • pip install --no-cache-dir mpi4py”
8.	Install python “setuptools” to properly configure the correct python setup tools needed to setup and install ESMF packages for the anaconda environment.This step is only necessary if you are manually installing/linking ESMF libraries to your Python environment on a supercomputer:
  • pip install setuptools==58.2.0
9.	 Install netcdf4 libraries to be able to load NWM domain files and meteorological datasets to execute the NWMv3.0 Forcings Engine: 
  • pip install --no-cache-dir netCDF4
10.	Install scipy libraries to load up interpolation methods that are utilized within the NextGen Forcings Engine:
  • pip install scipy
11.	cd to esmf-release-VERSION/src/addon/ESMPy directory. This is where the setup.py python script is located to install ESMF libraries onto your anaconda environment. Execute the following steps to link the ESMPy Python module to your anaconda environment:
  • Export ESMFMKFILE=/pathway/to/directory/esmf-release-8.1.0/lib/libO/Linux.intel.64.intel.default/esmf.mk
  • pip install .
12.	ESMF packages should be properly installed and linked to the intel compiler libraries for MPI capabilities. Open a python command line and “import ESMF” to make sure that the ESMF libraries were properly install on the anaconda environment. If it succeeds, then you should be good to go!

# NextGen Forcings Engine Basic Model Interface Setup and Execution
1.	Within your python environment, make sure to install the “bmipy” and “yaml” libraries to enable BMI functionality for the NextGen Forcings Engine to utilize. 
2.	Within the “NextGen_Forcings_Engine_BMI” directory, there is a sub-directory called “BMI_NextGen_Configs” that contains all the BMI configuration files needed for a Medium range forecast ran by GFS that would support a gridded model (./NextGen_Forcings_Engine_BMI/BMI_NextGen_Configs/Medium_Range/Gridded/config.yml), an unstructured mesh coastal model (./NextGen_Forcings_Engine_BMI/BMI_NextGen_Configs/Medium_Range/gridded/config.yml), and a hydrofabric unstructured mesh ((./NextGen_Forcings_Engine_BMI/BMI_NextGen_Configs/Medium_Range/hydrofabric/config.yml) as a supported domain to extract regridded forcings in a BMI-complaint fashion. Copy over one of those config.yml files to the “NextGen_Forcings_Engine_BMI” main directory for BMI execution of the NextGen Forcings Engine. Inside each config.yml contains a set of standard NWMv3.0 Forcing Engine variables that allow a given user to utilize a variety of methods, which are further described in the “./NextGen_Forcings_Engine_BMI/BMI_NextGen_Configs/README.md” file.
3.	To test out the NextGen Forcings Engine BMI functionality, please see the “./NextGen_Forcings_Engine_BMI/README.md” file that highlights the utility of each of the BMI Python scripts as well as supporting sub-directories contained within the repository. 

# Overview bullet points for modifying the original NWM Forcings Engine into a BMI complaint NextGen Forcings Engine capable of handling any domain types
  •	To streamline the NWMv3.0 Forcings engine into a Basic Model Interface application, we’ve had to initialize the BMI model using the same approach highlighted in the “genForcing.py” module and then directly reconfigure the forecast module (forecastMod.py) workflow to streamline the ability to update and produce gridded forcings for the WRFHydro domain based on a specified time stamp within the standard BMI functionality (“model_update_until”). This is all completed within the “model.py” module, which essentially mimics the “forecastMod.py” module within the “core” directory as a BMI-compliant module. 

  •	Once source code modifications were implemented, we were able to demonstrate the ability for the NWMv3.0 Forcings Engine to advertise gridded and unstructured mesh forcings across the CONUS WRFHydro domain back to the NextGen model engine.

  •	We also optimized the NWMv3.0 Forcings Engine source code within the BMI to bypass I/O functionality for producing netcdf forcing files and clearing data production within its scratch directory. 


# NextGen Forcings Engine Initialization Phase Workflow Overview
1.	 Read in NWM configuration file (config.py): Initialize “ConfigOptions” class, read in configuration file and assign respective class variables directory pathways, configuration type, and user input/output options.
2.	Read in “nwm_version” and “nwm_config” arguments and assign class variables for output directory information with Log file and output files (forcing engine version and operational configuration used for script execution). 
3.	Initialize MPI communications (parallel.py) class variables and then assign the MPI information (MPI_COMM_WORLD) for class variables (MPI rank, MPI size).
4.	Initialize WRF-Hydro geospatial object (geoMod.py), which contains class information about the modeling domain, local processor grid boundaries, and ESMF grid objects/fields to be used in regridding. The domain netcdf data is first opened, reads in coordinates and grid spacing, broadcasts data to processors which can then initialize their ESMF grid, and then reads internal domain netcdf data (including calculating slope grids) to be scattered accordingly to processors within MPI communicator.
5.	The WRF-Hydro geospatial object then reads in coordinate reference system/coordinates geospatial metadata and coordinates ONLY to main process (rank 0), which can be used for netcdf output files later on when they are created for the given user. 
6.	Initializes output object (ioMod.py), which creates class variables and “local slabs” to hold output grids for each given processor (ny_local, nx_local).
7.	Initializes input object (forcingInputMod.py) input forcing classes. These objects will contain information about our source products (I.E. data type, grid sizes, etc). Information will be mapped based on options specified by the user. In addition, input ESMF grid objects will be created to hold data for downscaling and regridding purposes. Essentially, each processor will initialize their own empty arrays for forcings variables to get regridded/filled and then bias correct/downscale based on the given operational configuration. 
8.	If we have a specified supplementary precipitation dataset for a given operational configuration, then we must initialize the supplementary precip (suppPrecipMod.py) class, which will process the precipitation dataset configurations, its regridding technique, and the file type/variable inputs for a given supplementary dataset.
9.	With all classes initialized for the Forcing Engine, we can now call the forecast module (forecastMod.py) to process the operational configuration and forecast data into regridded hourly output files configured to the NWM domain coordinate reference system and metadata.
# NextGen Forcings Engine Workflow to Process Operational Forecast Data Overview
1.	If user selected supplementary precipitation option for forcing dataset, then the script calls the module to provide disaggregation functionality (dissaggregateMod.py). This module reads in user configuration options for the supplementary precipitation data and then determines whether or not if the precipitation data is hourly or 6-hourly intervals. If the data is indeed 6-hourly accumulated precipitation, then the module will return the disaggregate function, which will essentially read the data and interpolate the precipitation data down to hourly intervals and set the forcing precipitation input array for ESMF regridding in the following steps.
2.	Calculate forecast cycle number, create output forecast directory and log file for forecast cycle (if needed) and configure AnA cycle for look back period based on operational configuration and current forecast cycle time (if needed).
3.	Loop through each output timestep
  a.	Reset final grids to missing values, get current timestep, output timestep, and previous timestep as well as operational configuration flags.
  b.	Loop over each of the input forcing specified, find the previous and next input cycle files, and regrid the forcing variable to the final output grids for a given timestep (forcingInputMod.py). For timesteps that require interpolation, two sets of input forcing grids will be regridded if we have come across new files and the process flag has been reset (regrid.py)
  c.	The given grib2 file is converted to a netcdf file and then regridded based on the predefined ESMF object for the NWM grid. Precipitation data is then converted from average precipitation rates to instantaneous precipitation rates for a given dataset through time interpolation objects (time_handling.py). Set any pixel cells outside the input domain to the global missing value.
  d.	Run temporal interpolation on the grids down to the NWM hourly forcing scale required, followed by running bias correction and downscaling grid objects respectively if applicable for a given operational dataset. 
  e.	Layer in regridded forcings for the given meteorological produce to expected output netcdf format and file configuration.
  f.	If there is a supplemental precipitation dataset available, then calculate the neighboring precipitation files to use (previous, next) for the given output timestep, regrid the supplemental precipitation data based on its ESMF regrid object, call “disaggregate_fun” (dissagregateMod.py) to disaggregate 6hr precip data to 1hr precip data (if necessary), run temporal interpolation on the grids, and relayer (replace) precipitation forcing output on the netcdf output object already generated for given timestep.
4.	Once the forecast cycle has been completed; close the log file and loop through the next forecast cycle number if necessary for given operational configuration. 
5.	If forecast cycle is complete then the NWMv3.0 Forcings Engine finalizes the log file and shuts down the MPI communications. 

# Testing BMI Python modules.
 - ./NextGen_Forcings_Engine/model.py: This file is the "NextGen Forcings Engine Driver" it takes inputs and gives an output
 - ./NextGen_Forcings_Engine/bmi_model.py: This is the Basic Model Interface that talks with the model. This is the NextGen Forcings Engine BMI class.
 - run_bmi_model.py: This is a file that mimics the framework, in the sense that it initializes the model with the BMI function. Then it runs the model with the BMI Update function, etc.
 - run_bmi_unit_test.py: This is a file that runs each BMI unit test to make sure that the BMI is complete and functioning as expected.
 - config.yml: This is a configuration file that the BMI reads to set inital_time (initial value of current_model_time) and all the required variables needed to drive the NextGen Forcings Engine.
 - environment.yml: Environment file with the required Python libraries needed to run the model with BMI and the NextGen Forcings Engine. Create the environment with this command: `conda env create -f environment.yml`, then activate it with `conda activate bmi_test`
 - slurm.job: Example Python submission script for a slurm job manager that is utilized to execute a MPI Python script (NextGen Forcings Engine) on the NOAA RDHPCS Hera supercomputer.
 - ./NextGen_Forcings_Engine/core/: The sub-directory containing all the support Python modules required to execute the NextGen Forcings Engine driver within model.py script.
 - ./BMI_NextGen_Configs/: A sub-directory containing all the current config.yml BMI file setupts that will allow a a user to driver the NextGen BMI Forcings Engine for a gridded domain, a coastal-model unstructured mesh domain, and a NextGen hydrofabric (VPU 06) unstructured mesh domain. 
 - ./NextGen_Domains/: A sub-directory that contains a sample domain geogrid file for a gridded domain, a coastal-model unstructured mesh domain, and a NextGen hydrofabric (VPU 06) unstructured mesh domain that can all be utitlized currently to regrid forcings for a Medium Range forecast simulation driver by GFS forcings.
 - ./NWM_Params/: An empty sub-directory that will eventually contain supporting climatology files that will support implementing various bias calibration and downscaling technqius that are only associated with the original WRF-Hydro domain (aka NOAH-OWP Modular).
 - ./Unit_Test_Output/: An sub-directory that is essentially the current scratch directory setup within each of the config.yml files to temporary place I/O commands and the log file of the progress of the NextGen Forcings Engine

# About
This is an implementation of a Python-based model that fulfills the Python language BMI interface and can be used in the Framework. It is intended to serve as a control for testing purposes, freeing the framework from dependency on any real-world model in order to test BMI related functionality.

## Test the complete BMI functionality
`python run_bmi_unit_test.py`

## Run the model
`python run_bmi_model.py`

## Sample output
'model time', 'U2D_ELEMENT', 'V2D_ELEMENT', 'LWDOWN_ELEMENT','SWDOWN_ELEMENT','T2D_ELEMENT','Q2D_ELEMENT','PSFC_ELEMENT','RAINRATE_ELEMENT' 
3600 -0.34978889998563434 1.9576997889275944 353.8881030867496 635.9115803301148 299.8541540030717 0.01014856288865672 101253.80092868667 0.0
7200 -0.55664622087895 1.7230866051531413 351.70806444209256 446.07219911115914 299.7995647600879 0.01005461030680388 101259.55711493774 0.0
10800 -0.2516353004284137 1.4639377367413153 347.3662126755073 221.91463406286712 299.4248073664293 0.01016205978429921 101303.53098399537 0.0
