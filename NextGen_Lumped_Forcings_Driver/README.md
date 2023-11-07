Required Python packages to execute lumped forcings driver:

1. pybind11
2. xarray
3. dask
4. netCDF4
5. numpy
6. gdal
7. re
8. geopandas
9. pandas
10. multiprocessing
11. wget
12. scipy
13. shapely
14. ssl
15. pyarrow
16. exactextract (python bindings can be install by downloading python packages and following instructions on GitHub https://github.com/jdalrym2/exactextract/tree/coverage-fraction-pybindings to directly link the ExactExtract python bindings to your python executable or anaconda environment)

###################### Python executable sample code for executing lumped forcings driver #############################
from NextGen_lumped_forcings_driver import NextGen_lumped_forcings_driver
NextGen_lumped_forcings_driver("/pathway/to/lumped_forcings/output",start_time=None, end_time=None, met_dataset="HRRR",hyfabfile="/pathway/to/NextGen_hydrofabric_geopackage",hyfabfile_parquet=None, met_dataset_pathway="/pathway/to/data/source",weights_file=None,netcdf=False,csv=True,bias_calibration=False,downscaling=False,CONUS=False,AnA=False,num_processes=1)
######################################################################################################################

The NextGen lumped forcings driver takes the following inputs:

1. output_root (string) = Pathway where NextGen netcdf/csv files and scratch data will be genereated. (Required -> AORC, GFS, CFS, HRR)

2. start_time (string, default=None) = A string ('YYYY-MM-DD HH:00:00') indicating the start time for forecast/reanalysis data needed. This is only needed for AORC reanlysis data or HRRR AnA forecast configuration. GFS and CFS data simply produces their respective operational configurations beginning at the start of their forecast cycle. (Required -> AORC, HRRR)

3. end_time (string, default = None) = A string ('YYYY-MM-DD HH:00:00') indicating the end time for forecast/reanalysis data needed. This is only needed for AORC reanlysis data. HRRR, GFS, and CFS data simply produces their respective operational configurations ending at their expected time period (HRRR ~ 18-48 hour forecast, GFS ~ 10 day forecast, CFS ~ 30 day forecast). (Required -> AORC)

4. met_dataset (string) = A string indicating which meteorological dataset that is requested by the user (options = 'AORC', 'GFS', 'HRRR', 'CFS'). (Required -> AORC, GFS, CFS, HRRR)

5. hyfabfile (string) = File pathway that points to the user specified NextGen hydrofabric geopackage file, which will be used for the their NextGen lumped formulation that they need forcings for. (Required -> AORC, GFS, CFS, HRRR)

6. hyfabfile_parquet (string, default=None) = File pathway that points to the user specified NextGen hydrofabric VPU/CONUS parquet file containing the required forcing metadata to implement NCAR bias calibration and downscaling functions. This is within hydrofabric version 2.0 repository and newer versions as well moving forward. (Optional -> GFS, CFS, HRRR)
  
7. met_dataset_pathway (string, default=None) = Pathway where forecast dataset is located to generate lumped forcings for AORC, GFS, HRRR, and CFS data. This is required when you specify data for one of these datasets. (Required -> GFS, CFS, HRRR, Optional for AORC if you're on the NWC servers where you can directly connect to the ERRDAP server and extract data, otherwise it's required then for AORC data)
(AORC, GFS, and CFS modules require direct pathway to the netcdf (AORC) or grib2 files (GFS, CFS) for the given time span or forecast cycle you wish to produce files for. HRRR module requires pathway to the HRRR directory encapulating multiple HRRR forecast cycles (e.g. hrrr.20230104  hrrr.20230105) contained within that data directory for Short range and AnA configuration to properly assimilate the data)

8. weights_file (string, default=None) = Pathway to where a given user has already produced the ExactExtract coverage fraction weights csv file for the current hydrofabric dataset they are inputing into the module from a previous run and instead would like the option to just feed it into the module and bypass the weights production step to optimize the file production runtime for a given meteorological forcing dataset

9. netcdf (boolean flag, default=True) = Python boolean flag (True, False) indicating whether the user wants a NextGen formatted netcdf file produce for the meteorological dataset. (Required if csv=False)

10. csv (optional, boolean flag, default=False) = Python boolean flag (True, False) indicating whether the user wants NextGen formatted csv catchment files produced for the meteorological dataset. (Required if netcdf=False)
(Warning, modules have been modified to not accept this argument if the input user has specified as using a CONUS geopackage file (CONUS=True). This is becuase the I/O file memory will generally break a system when producing 800,000+ csv files at once.)

11. bias_calibration (boolean flag, default=False) = Python boolean flag (True, False) indicating whether the user wants to account for NCAR bias calibration techniques previously translated in the National Water Model (NWM) Forcings Engine for operational forecast datasets (Optional -> GFS, CFS, HRRR).

12. downscaling (boolean flag, default=False) = Python boolean flag (True, False) indicating whether the user wants to account for NCAR downscaling techniques previously translated in the National
Water Model (NWM) Forcings Engine for operational forecast datasets (Optional -> GFS, CFS, HRRR).

13. CONUS (boolean flag, default=False) = Python boolean flag (True, False) indicating whether the user NextGen hydrofabric file contains a subset of catchments, or the entire CONUS network of catchments. This information is only required for requesting an AORC dataset. The python environment is constrained to the servers memory allocation, which will vary from platform to platform. (Required -> AORC)
(Warning; Requesting a long-term AORC dataset containing the entire CONUS network of catchments within the hydrofabric file may break the script due to memory allocation issues on your given environment! Make sure you have sufficent memory allocation (>15-20 GBs) on your environment for the CONUS.gpkg file creation)

14. AnA (boolean flag, default=False) = Python boolean flag (True, False) indicating whether or not the user is requesting an 'Analysis and Assimilation' operational configuration for the HRRR forecast data. This will produce lumped forcings for a 28-hour look back period (hourly frequency -28,..., -2, -1, 0-hour lookback) for hour number one (f01) within a given HRRR forecast cycle. This option is only valid for the HRRR meteoroloigcal forcing dataset. (Required -> HRRR)

15. num_proccesses (integer, default=1) = The number of proccessors available to the Python lumped forcings driver. These python scripts utilize multi-threading procedures, which helps to speed up the production of the NextGen lumped forcings dataset.
(Warning; requesting too many threads on  given environment may cause the system to run out of memory quickly as the script can only handle so many forcing files at once.)

The given user should utilize sample function calls for each forcings dataset available in the lumped forcings driver. The examples are shown within the Run_NextGen_lumped_driver.py script available in the repository.

