# NextGen Forcings Engine Forcing Engine Configuration File

Input options to the forcing engine include:
1. Choices for input forcing files to use.
2. Options for specifying date ranges and forecast intervals for input files.
3. Choices for ESMF regridding techniques.
4. Choices for optional downscaling techniques.
5. Choices for optional bias correction techniques.
6. Choices for optional supplemental precipitation products.
7. Choices for optional ensemble member variations.
8. Choices for output directories to place final output files.
9. Choice for a single output file containing entire timeseries of regridded data for given domain

### time_step_seconds
This variable is preset at 3600 seconds, which just reflects the BMI interval in which we advertise meteorological forcings to the framework. As of now, this value will not change until both the framework and the forcing engine adjust regridding intervals
- Example- time_step_seconds: 3600

### initial_time
This variable will always be set to 0, which just tells the framework that we're starting at the reference date stated in the config.yml file here below, which is labeled as "RefcstBDateProc"
- Example-  initial_time: 0

### NWM_VERSION
This is just a configuruation option telling the Forcing Engine BMI what version of the NWM this is forcing. This variable can be used for netcdf metadata output as a reference to the user
- Example- NWM_VERSION: 4.0 

### NWM_CONFIG
This variable just refers to a string of the Forcing Engine BMI configuration the user is implementing here. This variable can be used for netcdf metadata output as a reference to the user
- Example- NWM_CONFIG: "NWMv4_Medium_Range"

### InputForcings 
Choose a set of value(s) of forcing variables to be processed for WRF-Hydro. Please be advised that the order of which the values are chosen below are the order that the final products will be layered into the final LDASIN files. See documentation for additional information and examples. The following is a global set of key values to map forcing files to variables within LDASIN files for WRF-Hydro. The forcing engine will map files to external variable names internally. For custom external native forcing files (see documenation), the code will expect a set of named variables to process. The following is a mapping of numeric values to external input native forcing files:
1. NLDAS GRIB retrospective files
2. NARR GRIB retrospective files
3. GFS GRIB2 Global production files on the full gaussian grid
4. NAM Nest GRIB2 Conus production files
5. HRRR GRIB2 Conus production files
6. RAP GRIB2 Conus 13km production files
7. CFSv2 6-hourly GRIB2 Global production files
8. WRF-ARW - GRIB2 Hawaii nest files
9. GFS GRIB2 Global production files on 0.25 degree lat/lon grids.
10. Custom NetCDF hourly forcing files
11. Custom NetCDF hourly forcing files
12. AORC
13. Hawaii 3-km NAM Nest.
14. Puerto Rico 3-km NAM Nest.
15. Alaska 3-km Alaska Nest
16. NAM_Nest_3km_Hawaii_Radiation-Only
17. NAM_Nest_3km_PuertoRico_Radiation-Only
18. WRF-ARW GRIB2 PuertoRico
19. HRRR Alaska GRIB2 production files
20. Alaska Analysis and Assimilation
21. AORC Alaska
22. Alaska Extended Analysis and Assimilation
23. ERA5-Interim
24. National Blended Models
25. National Digitial Forecast Database
- Example- InputForcings: [3,25]

### InputForcingDirectories
Specify the input directories for each forcing product.
- Example- InputForcingDirectories: [./GFS,./NDFD]

### InputForcingTypes
Specify the file type for each forcing (comma separated). Valid types are GRIB1, GRIB2, NETCDF, and NETCDF4
- Example- InputForcingTypes: [GRIB2,GRIB2]

### InputMandatory
Specify whether the input forcings listed above are mandatory, or optional. This is important for layering contingencies if a product is missing, but forcing files are still desired. 0 - Not mandatory, 1 - Mandatory. NOTE!!! If no files are found for any products, code will error out indicating the final field is all missing values.
- Example- InputMandatory: [1,1]

### OutputFrequency
Specify the output frequency in minutes. Note that any frequencies at higher intervals than what if provided as input will entail input forcing data being temporally interpolated.
- Example- OutputFrequency: 60

### SubOutputHour
New variable currently for NWMv3.1 operations to properly ingest GFS 13km forecast data that outputs various frequencies throughout the forecast cycle lifetime. This variable will properly account for reading time slices of the forecast cycle. Currently only needed for GFS 13km operational configuration. Otherwise, set this value to 0.
- Example- SubOutputHour: 0

### SubOutputFreq
New variable currently for NWMv3.1 operations to properly ingest GFS 13km forecast data that outputs various frequencies throughout the forecast cycle lifetime. This variable will properly account for reading time slices of the forecast cycle based on frequency of occurence. Currently only needed for GFS 13km operational configuration. Otherwise, set this value to 0.
- Example- SubOutputFreq: 0  

### ScratchDir
Specify a scratch directory that will be used for storage of temporary files. These files will be removed automatically by the program. at the end of the BMI instance. However, this directory will also store the output forcing file if requested by the user as well (will not be deleted in this instance).
- Example- ScratchDir: "./ScratchDir"

### Output
Specify whether or not you would like to request a single netcdf output file containg all of the regridded meteorological forcing fields for the domain configuration you set up within the config.yml file. Output: 0=No, 1=Yes
- Example- Output: 1

### compressOutput
Flag to activate scale_factor / add_offset byte packing in the output files. 0 - Deactivate compression 1 - Activate compression, Only applicable in this instance when you request a netcdf output forcing file (Output: 1). Otherwise, just set to 0.
- Example- compressOutput: 0

### floatOutput
Flag to use floating point output vs scale_factor / add_offset byte packing in the output files (the default). 0 - Use scale/offset encoding, 1 - Use floating-point encoding. Only applicable in this instance when you request a netcdf output forcing file (Output: 1). Otherwise, just set to 0.
- Example- floatOutput: 0

### AnAFlag
If this is AnA run, set AnAFlag to 1, otherwise 0. Setting this flag will change the behavior of some Bias Correction routines as the ForecastInputOffsets options.
- Example- AnAFlag: 1

### LookBack
Specify a lookback period in minutes to process data. This is required if you are processing a restrospective dataset or an AnA operational configuration. This value should specify how far back you need to look in time from your "RefcstBDateProc" start date that you specified. In this instance, that start date will be your actual end date. If no LookBack specified, please specify -9999.
- Example- LookBack: 180

### RefcstBDateProc
If running an operational configuration in realtime, this will be the defined start date for the NextGen Forcing Engine BMI which is assumed to be the beginning of the forecast cycle (i.e. hour 0). From there the first time step will be hour 1 from the start date specified here. If you're running an AnA or a retrospective dataset however, this variable becomes the end date of the simulation and the "LookBack" value specified above will be how far back you look in time for the AnA or retrospective dataset.
- Example- RefcstBDateProc: 202210071400

### ForecastFrequency
Specify a forecast frequency in minutes. This value specifies how often to generate a set of forecast forcings. If generating hourly retrospective forcings, specify this value to be 60.
- Example- ForecastFrequency: 60

### ForecastShift
Forecast cycles are determined by splitting up a day by equal ForecastFrequency interval. If there is a desire to shift the cycles to a different time step, ForecastShift will shift forecast cycles ahead by a determined set of minutes. For example, ForecastFrequency of 6 hours will produce forecasts cycles at 00, 06, 12, and 18 UTC. However, a ForecastShift of 1 hour will produce forecast cycles at 01, 07, 13, and 18 UTC. NOTE - This is only used by the realtime instance to calculate forecast cycles accordingly. Re-forecasts will use the beginning and ending dates specified in conjunction with the forecast frequency to determine forecast cycle dates.
- Example- ForecastShift: 0

### ForecastInputHorizons
Specify how much (in minutes) of each input forcing is desires for each forecast cycle. See documentation for examples. The length of this array must match the input forcing choices.
- Example- ForecastInputHorizons: [60, 60]

### ForecastInputOffsets
This option is for applying an offset to input forcings to use a different forecasted interval. For example, a user may wish to use 4-5 hour forecasted fields from an NWP grid from one of their input forcings. In that instance the offset would be 4 hours, but 0 for other remaining forcings.
- Example- ForecastInputOffsets: [0, 0]

### GeogridIn
Specify a geogrid file (e.g. latitude, longitude, mesh connectivity, elevation, slope) that defines domain to which the forcings are being processed to.
- Example- GeogridIn: ./geo_em_CONUS.nc

### SpatialMetaIn
Specify the optional land spatial metadata file. If found, coordinate projection information and coordinate will be translated from to the final output file. This variable is only a special case if the user is specifying the original WRF-Hydro domain from earlier NWM versions. Otherwise, just leave the one blank ('')
- Example- SpatialMetaIn: ./GEOGRID_LDASOUT_Spatial_Metadata_CONUS.nc

### GRID_TYPE
This tells the NextGen Forcings Engine BMI which grid type the engine is initalizing as a BMI instance. This is a required field and the proper string values should be "gridded", "hydrofabric", or "unstructured".
- Example- GRID_TYPE: "gridded"

### LONVAR
This variable is the naming convention of the longitude variable within the "GeogridIn" file the user has specified. Variable naming convention ONLY for gridded domain configurations. This is required so the NextGen Forcings Engine BMI can dyanmically initialize the domain geogrid as an ESMF regridding object.
- Example- LONVAR: "XLONG_M"

### LATVAR
This variable is the naming convention of the latitude variable within the "GeogridIn" file the user has specified. Variable naming convention ONLY for gridded domain configurations. This is required so the NextGen Forcings Engine BMI can dyanmically initialize the domain geogrid as an ESMF regridding object.
- Example- LATVAR: "XLAT_M"

### HGTVAR
This variable is the naming convention of the elevation variable (describes the elevation estimate for each grid cell or mesh node for unstructured mesh) within the "GeogridIn" file the user has specified. This is optional for the NextGen Forcings Engine BMI ONLY if the user did not specify a downscaling method. If the user specifies a bias calibration or a downscaling method however, then this variable becomes mandatory and the NextGen Forcings Engine BMI will throw an error if this netcdf variable naming convention is not specified in the "GeogridIn" file. 
- Example- HGTVAR: "Elevation"

### HGTVAR_ELEM
This variable is the naming convention of the elevation element variable (describes the elevation estimate for mesh element ONLY in an unstructured mesh) within the "GeogridIn" file the user has specified. This is optional for the NextGen Forcings Engine BMI ONLY if the user did not specify a downscaling method. If the user specifies a bias calibration or a downscaling method however, then this variable becomes mandatory and the NextGen Forcings Engine BMI will throw an error if this netcdf variable naming convention is not specified in the "GeogridIn" file. 
- Example- HGTVAR_ELEM: "Elevation_Element"

### SLOPE
This variable is the naming convention of the slope variable (describes the slope estimate for each grid cell or mesh element) within the "GeogridIn" file the user has specified. This is optional for the NextGen Forcings Engine BMI ONLY if the user did not specify a downscaling method. If the user specifies a bias calibration or a downscaling method however, then this variable becomes mandatory and the NextGen Forcings Engine BMI will throw an error if this netcdf variable naming convention is not specified in the "GeogridIn" file. 
- Example- SLOPE: "Slope"

### SLOPE_AZIMUTH
This variable is the naming convention of the slope azmuith variable (describes the slope tilt estimate for each grid cell or mesh element) within the "GeogridIn" file the user has specified. This is optional for the NextGen Forcings Engine BMI ONLY if the user did not specify a downscaling method. If the user specifies a bias calibration or a downscaling method however, then this variable becomes mandatory and the NextGen Forcings Engine BMI will throw an error if this netcdf variable naming convention is not specified in the "GeogridIn" file. 
- Example- SLOPE_AZIMUTH: "Slope_Tilt"

### SINALPHA
This variable is the naming convention of the sine angle of the grid cell slope variable (describes the slope tilt estimate for each grid cell or mesh element) within the "GeogridIn" file the user has specified. As of now, this variable is ONLY assumed to be in the original WRF-Hydro "GeogridIn" file from the older NWM versions OR from a gridded domain configuration that has the same calculated variable. This is optional for the NextGen Forcings Engine BMI ONLY if the user did not specify a downscaling method. If the user specifies a bias calibration or a downscaling method however, then this variable becomes mandatory and the NextGen Forcings Engine BMI will throw an error if this netcdf variable naming convention is not specified in the "GeogridIn" file. 
- Example- SINALHPA: "SINALPHA"

### COSALPHA
This variable is the naming convention of the cosine angle of the grid cell slope variable (describes the slope tilt estimate for each grid cell or mesh element) within the "GeogridIn" file the user has specified. As of now, this variable is ONLY assumed to be in the original WRF-Hydro "GeogridIn" file from the older NWM versions OR from a gridded domain configuration that has the same calculated variable. This is optional for the NextGen Forcings Engine BMI ONLY if the user did not specify a downscaling method. If the user specifies a bias calibration or a downscaling method however, then this variable becomes mandatory and the NextGen Forcings Engine BMI will throw an error if this netcdf variable naming convention is not specified in the "GeogridIn" file. 
- Example- COSALHPA: "COSALPHA"

### NodeCoords
This variable is the naming convention of the node coordinates variable within the "GeogridIn" file the user has specified for ONLY an unstructured mesh or the NextGen hydrofabric. This is a 2-D array stating the latitude and longitude coordinates for all the nodes in the mesh. This is required so the NextGen Forcings Engine BMI can dyanmically initialize the domain geogrid as an ESMF regridding object.
- Example- NodeCoods: "nodecoords"

### ElemCoords
This variable is the naming convention of the element coordinates variable within the "GeogridIn" file the user has specified for ONLY an unstructured mesh or the NextGen hydrofabric. This is a 2-D array stating the latitude and longitude coordinates for all the elements in the mesh. This is required so the NextGen Forcings Engine BMI can dyanmically initialize the domain geogrid as an ESMF regridding object.
- Example- ElemCoods: "elemcoords"

### ElemConn
This variable is the naming convention of the element connectivity variable within the "GeogridIn" file the user has specified for ONLY an unstructured mesh or the NextGen hydrofabric. This is a 2-D array stating the node ids for each element connecting the entire mesh structure. This is required so the NextGen Forcings Engine BMI can dyanmically initialize the domain geogrid as an ESMF regridding object.
- Example- ElemConn: "elemconn"

### NumElemConn
This variable is the naming convention of the number of nodes per element variable within the "GeogridIn" file the user has specified for ONLY an unstructured mesh or the NextGen hydrofabric. This is a 1-D array stating the how many nodes are connecting each element within the unstructured mesh. This is required so the NextGen Forcings Engine BMI can dyanmically initialize the domain geogrid as an ESMF regridding object.
- Example- NumElemConn: "numelemconn"

### ElemID
This variable is the naming convention of the element id variable within the "GeogridIn" file the user has specified for ONLY the NextGen hydrofabric. This is a 1-D array stating the catchment id numeric naming convention within the "divides" geopackage layer of a given NextGen hydrofabric file. This variable is required in order for the NextGen Forcings Engine to properly advertise the element ids of the unstructured mesh linked to the NextGen hydrofabric catchment ids. 
- Example- ElemID: "element_ids"

### IgnoredBorderWidths
Specify a border width (in grid cells) to ignore for each input dataset. NOTE: generally, the first input forcing should always be zero or there will be missing data in the final output
- Example- IgnoredBorderWidths: [0,10]

### RegridOpt
Choose regridding options for each input forcing files being used. Options available are: 1 - ESMF Bilinear, 2 - ESMF Nearest Neighbor, 3 - ESMF Conservative Bilinear
- Example- RegridOpt: [1,1]

### ForcingTemporalInterpolation
Specify an temporal interpolation for the forcing variables. Interpolation will be done between the two neighboring input forcing states that exist. If only one nearest state exist (I.E. only a state forward in time, or behind), then that state will be used as a "nearest neighbor". NOTE - All input options here must be of the same length of the input forcing number. Also note all temporal interpolation occurs BEFORE downscaling and bias correction. 0 - No temporal interpolation. 1 - Nearest Neighbor, 2 - Linear weighted,  average
- Example- ForcingTemporalInterpolation: [0,0]

### TemperatureBiasCorrection
Specify a temperature bias correction method. 0 - No bias correction, 1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY, 2 - Custom NCAR bias-correction based on HRRRv3 analysis - based on hour of day (USE WITH CAUTION), 3 - NCAR parametric GFS bias correction, 4 - NCAR parametric HRRR bias correction
- Example- TemperatureBiasCorrection: [0, 4]

### PressureBiasCorrection
Specify a surface pressure bias correction method. 0 - No bias correction, 1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY
- Example- PressureBiasCorrection: [0,0]

### HumidityBiasCorrection
Specify a specific humidity bias correction method. 0 - No bias correction, 1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY, 2 - Custom NCAR bias-correction based on HRRRv3 analysis - based on hour of day (USE WITH CAUTION).
- Example- HumidityBiasCorrection: [0,0]

### WindBiasCorrection
Specify a wind bias correction. 0 - No bias correction, 1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY, 2 - Custom NCAR bias-correction based on HRRRv3 analysis - based on hour of day (USE WITH CAUTION), 3 - NCAR parametric GFS bias correction, 4 - NCAR parametric HRRR bias correction
- Example- WindBiasCorrection: [0, 4]

### SwBiasCorrection
Specify a bias correction for incoming short wave radiation flux. 0 - No bias correction, 1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY, 2 - Custom NCAR bias-correction based on HRRRv3 analysis (USE WITH CAUTION).
- Example- SwBiasCorrection: [0, 2]

### LwBiasCorrection
Specify a bias correction for incoming long wave radiation flux. 0 - No bias correction, 1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY, 2 - Custom NCAR bias-correction based on HRRRv3 analysis, blanket adjustment (USE WITH CAUTION), 3 - NCAR parametric GFS bias correction
- Example- LwBiasCorrection: [0, 2]

### PrecipBiasCorrection
Specify a bias correction for precipitation. 0 - No bias correction, 1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY
- Example- PrecipBiasCorrection: [0, 0]

### TemperatureDownscaling
Specify a temperature downscaling method: 0 - No downscaling, 1 - Use a simple lapse rate of 6.75 degrees Celsius to get from the model elevation to the WRF-Hydro elevation, 2 - Use a pre-calculated lapse rate regridded to the WRF-Hydro domain (only NWM), 3 - Use a dynamic lapse rate calculated at each timstep
- Example- TemperatureDownscaling: [3, 3]

### PressureDownscaling
Specify a surface pressure downscaling method: 0 - No downscaling, 1 - Use input elevation and WRF-Hydro elevation to downscale surface pressure.
- Example- PressureDownscaling: [1, 1]

### ShortwaveDownscaling
Specify a shortwave radiation downscaling routine. 0 - No downscaling, 1 - Run a topographic adjustment using the WRF-Hydro elevation
- Example- ShortwaveDownscaling: [1, 1]

### PrecipDownscaling
Specify a precipitation downscaling routine. 0 - No downscaling, 1 - NWM mountain mapper downscaling using monthly PRISM climo.
- Example- PrecipDownscaling: [0, 0]

### HumidityDownscaling
Specify a specific humidity downscaling routine. 0 - No downscaling, 1 - Use regridded humidity, along with downscaled temperature/pressure to extrapolate a downscaled surface specific humidty.
- Example- HumidityDownscaling: [1, 1]

### DownscalingParamDirs
Specify the input parameter directory containing necessary downscaling grids. This is ONLY needed for the original NWM WRF-Hydro domain. Otherwise, just point it to a random directory and it will be ignored. 
- Example- DownscalingParamDirs: ["./forcingParam/AnA", "./forcingParam/AnA"]

### SuppPcp
Choose a set of supplemental precipitation file(s) to layer into the final LDASIN forcing files processed from the options above. The following is a mapping of numeric values to external input native forcing files:
1. MRMS GRIB2 hourly radar-only QPE
2. MRMS GRIB2 hourly gage-corrected radar QPE
3. WRF-ARW 2.5 km 48-hr Hawaii nest precipitation.
4. WRF-ARW 2.5 km 48-hr Puerto Rico nest precipitation.
5. CONUS MRMS GRIB2 hourly MultiSensor QPE (Pass 2 or Pass 1)
6. Hawaii MRMS GRIB2 hourly MultiSensor QPE (Pass 2 or Pass 1)
7. MRMS SBCv2 Liquid Water Fraction (netCDF only)
8. NBM Conus MR
9. NBM Alaska MR
10. Alaska MRMS (no liquid water fraction)
11. Alaska Stage IV NWS Precip
12. CONUS Stage IV NWS Precip
13. MRMS PrecipFlag precipitation classification file
- Example- SuppPcp: [1, 5, 13]

### SuppPcpForcingTypes
Specify the file type for each supplemental precipitation file (comma separated). Valid types are GRIB1, GRIB2, and NETCDF (GRIB files will be converted internally with WGRIB[2])
- Example- SuppPcpForcingTypes: [GRIB2, GRIB2, GRIB2]

### SuppPcpDirectories
Specify the correponding supplemental precipitation directories that will be searched for input files.
- Example- SuppPcpDirectories: ['./MRMS_CONUS_GAUGE', './MRMS_CONUS_MULTISENSOR', './MRMS_CLASSIFICATION']

### SuppPcpParamDir
Specify an optional directory that contains supplemental precipitation parameter fields, I.E monthly RQI climatology. This is ONLY needed for the original NWM WRF-Hydro domain. Otherwise, just point it to a random directory and it will be ignored. 
- Example- SuppPcpParamDir: ['./forcingParam/AnA','./forcingParam/AnA','./forcingParam/AnA']

### RegridOptSuppPcp
Specify regridding options for the supplemental precipitation products. Options available are: 1 - ESMF Bilinear, 2 - ESMF Nearest Neighbor, 3 - ESMF Conservative Bilinear
- Example- RegridOptSuppPcp: [1, 1, 1]

### SuppPcpTemporalInterpolation
# Specify the time interpretation methods for the supplemental precipitation products.
- Example- SuppPcpTemporalInterpolation: [0, 0, 0]

### SuppPcpInputOffsets
In AnA runs, this value is the offset from the available forecast and 00z. For example, if forecast are available at 06z and 18z, set this value to 6
- Example- SuppPcpInputOffsets = [0, 0, 0]

### SuppPcpMandatory
Specify whether the Supplemental Precips listed above are mandatory, or optional. This is important for layering contingencies if a product is missing, but forcing files are still desired. 0 - Not mandatory, 1 - Mandatory
- Example- SuppPcpMandatory: [0, 0, 0]

### RqiMethod
Optional RQI method for radar-based data. 0 - Do not use any RQI filtering. Use all radar-based estimates. 1 - Use hourly MRMS Radar Quality Index grids, 2 - Use NWM monthly climatology grids (NWM only!!!!)
- Example- RqiMethod: 2

### RqiThreshold
Optional RQI threshold to be used to mask out. Currently used for MRMS products. Please choose a value from 0.0-1.0. Associated radar quality index files will be expected from MRMS data.
- Example- RqiThreshold: 0.9

### cfsEnsNumber
Choose ensemble options for each input forcing file being used. Ensemble options include: CFS ensemble number ONLY. No other Forecast models use ensemble numbers as an operational configuration. Lagged ensemble operational configurations are just simply using older forecast simulation estimates and their respective ensembles as a single Forcing Engine BMI instance. This is ONLY specifying certain CFS ensemble configurations, which can directly partition out certain CFS ensemble members as the input for a given NextGen Forcings Engine BMI instance. 
- Example-  cfsEnsNumber = '1'

### custom_input_fcst_freq
These are options for specifying custom input NetCDF forcing files (in minutes). Choose the input frequency of files that are being processed. I.E., are the input files every 15 minutes, 60 minutes, 3-hours, etc. Please specify the length of custom input frequencies to match the number of custom NetCDF inputs selected above in the Logistics section.
- Example-  custom_input_fcst_freq: []

### includeLQFrac
Include LQFRAC variable (liquid fraction of precipitation). Enable if using HRRR, RAP, GFS, MRMS, or NDFD. 
- Example- includeLQFrac: 1
