# NextGen Forcings Engine Configuration Guide

This document describes all available input options within the `config.yml` configuration file used by the **NextGen Forcings Engine BMI**. 

Sample templates for standard National Water Model (NWM) operational configurations are located in the `BMI_NextGen_Configs/config_templates` directory.

---

## 1. System & Execution Settings

Configuration parameters defining runtime environments, execution flags, and metadata output.

| Parameter | Type | Default / Example | Description |
| :--- | :--- | :--- | :--- |
| `time_step_seconds` | Integer | `3600` | The BMI update interval (in seconds) used to advertise forcings to the framework. |
| `initial_time` | Integer | `0` | Initial time reference offset. Set to `0` to align with `RefcstBDateProc`. |
| `NWM_VERSION` | String | `"4.0"` | NWM version string populated in NetCDF output metadata. |
| `NWM_CONFIG` | String | `"NWMv4_Medium_Range"` | Configuration descriptor written to NetCDF output metadata. |
| `GRID_TYPE` | String | `"gridded"` | Target domain grid topology. Valid options: `"gridded"`, `"hydrofabric"`, or `"unstructured"`. |
| `project` | String | `"noaa_owp"` | Output schema compatibility branch. Valid options: `"noaa_owp"` or `"ngwpc"`. |
| `ScratchDir` | String | `"./ScratchDir"` | Temporary working directory for GRIB conversion and staging. NetCDF outputs are also written here if requested. |

---

## 2. Output & Compression Options

Settings controlling the generation and formatting of standalone NetCDF forcing files.

| Parameter | Options | Description |
| :--- | :--- | :--- |
| `Output` | `0` = No<br>`1` = Yes | Generates a single consolidated NetCDF file containing all regridded fields across the run duration. Set to `0` when running purely as a BMI provider. |
| `compressOutput` | `0` = Off<br>`1` = On | Enables `scale_factor` / `add_offset` byte packing in output NetCDF files (requires `Output: 1`). |
| `floatOutput` | `0` = Packed<br>`1` = Float | Forces output fields to be written as uncompressed 32-bit floating-point values (requires `Output: 1`). |
| `OutputFrequency` | Minutes | Target output timestep interval in minutes (e.g., `60`). Higher input frequencies trigger temporal interpolation. |
| `SubOutputHour` | Integer | Accounting parameter for variable-frequency outputs (e.g., GFS 13km forecast cycles). Default to `0`. |
| `SubOutputFreq` | Integer | Interval frequency parameter corresponding to `SubOutputHour`. Default to `0`. |
| `includeLQFrac` | `0` = Off<br>`1` = On | Includes liquid precipitation fraction (`LQFRAC`) in outputs. Supported for HRRR, RAP, GFS, MRMS, and NDFD. |

---

## 3. Spatial & Domain Specifications

Variables defining coordinate dimensions, variable mappings, and mesh connectivity in target domain files (`GeogridIn`).

| Parameter | Applicable Domain | Description |
| :--- | :--- | :--- |
| `Geopackage` | Hydrofabric | Path to NextGen hydrofabric GeoPackage (`.gpkg`). Required if `GeogridIn` ESMF mesh is not yet built. |
| `GeogridIn` | All Domains | Path to the ESMF-compliant NetCDF grid/mesh file containing domain spatial attributes. |
| `SpatialMetaIn` | Gridded (Legacy) | Path to optional land spatial metadata file (e.g., `GEOGRID_LDASOUT_Spatial_Metadata_CONUS.nc`). |
| `NWM_Geogrid` | Retrospective (#27) | Path to `geo_em_NWM_DOMAIN.nc` required to construct ESMF grid objects for NWM v3 retrospective runs. |
| `parquet` | Optional | Path to supplementary `.parquet` file containing elevation, slope, and azimuth data for downscaling. |
| `LONVAR` / `LATVAR` | Gridded | Variable names for longitude/latitude in `GeogridIn` (e.g., `"XLONG_M"`, `"XLAT_M"`). Must contain `dx`/`dy` spacing attributes for downscaling. |
| `HGTVAR` | All Domains | Elevation variable name in `GeogridIn` (e.g., `"Elevation"`). Mandatory if downscaling/bias correction is active. |
| `HGTVAR_ELEM` | Unstructured | Element elevation variable name for unstructured mesh geometries (e.g., `"Elevation_Element"`). |
| `SLOPE` / `SLOPE_ELEM` | All Domains | Terrain slope variable names for nodes/cells and unstructured elements (e.g., `"Slope"`). |
| `SLOPE_AZIMUTH` / `_ELEM` | All Domains | Terrain slope azimuth (tilt) variable names for nodes/cells and unstructured elements (e.g., `"Slope_Tilt"`). |
| `SINALPHA` / `COSALPHA` | Gridded (Legacy) | Sine/cosine grid rotation angle variable names from legacy WRF-Hydro geogrid files. |
| `NodeCoords` | Mesh / Hydrofabric | 2D array variable name specifying node latitude/longitude coordinates (e.g., `"nodecoords"`). |
| `ElemCoords` | Mesh / Hydrofabric | 2D array variable name specifying element center latitude/longitude coordinates (e.g., `"elemcoords"`). |
| `ElemConn` | Mesh / Hydrofabric | 2D array variable name defining element node connectivity indices (e.g., `"elemconn"`). |
| `NumElemConn` | Mesh / Hydrofabric | 1D array variable name stating the number of nodes per mesh element (e.g., `"numelemconn"`). |
| `ElemID` | Hydrofabric | 1D array variable name linking mesh elements to NextGen catchment IDs (e.g., `"element_ids"`). |

---

## 4. Input Forcing Configuration

Settings controlling input product selection, file locations, lookback horizons, and temporal mapping.

### Primary Input Parameters

| Parameter | Type / Format | Default / Example | Description |
| :--- | :--- | :--- | :--- |
| `InputForcings` | Integer Array | `[3, 25]` | Selects the forcing product IDs to process. Layering order in output files corresponds to array index order. |
| `InputForcingDirectories` | String Array | `["/data/GFS", "/data/NDFD"]` | Pathways for input datasets. Set to `""` for dynamic AWS S3 downloads (e.g., AORC `#12` or NWM Retrospective `#27`). |
| `InputForcingTypes` | String Array | `["GRIB2", "GRIB2"]` | Native file format per product in `InputForcings`. Valid options: `GRIB1`, `GRIB2`, `NETCDF`, `NETCDF4`. |
| `InputMandatory` | Binary Array | `[1, 1]` | `1` = Mandatory, `0` = Optional. Determines fallback behavior if forcing files are missing. |
| `IgnoredBorderWidths` | Integer Array | `[0, 10]` | Number of outer boundary grid cells to drop per input product (the primary product should be `0`). |
| `RegridOpt` | Integer Array | `[1, 1]` | ESMF regridding method per product: `1` = Bilinear, `2` = Nearest Neighbor, `3` = Conservative Bilinear. |
| `ForcingTemporalInterpolation` | Integer Array | `[0, 0]` | Pre-downscaling/bias-correction interpolation: `0` = None, `1` = Nearest Neighbor, `2` = Linear Weighted. |
| `ForecastInputHorizons` | Integer Array | `[60, 60]` | Duration (in minutes) of each forcing product utilized per forecast cycle. |
| `ForecastInputOffsets` | Integer Array | `[0, 0]` | Offset (in minutes) applied to force usage of specific forecast lead-time windows. |
| `custom_input_fcst_freq` | Integer Array | `[60]` | Input file frequency (in minutes) for custom NetCDF datasets (Products `#10` / `#11`). |
| `cfsEnsNumber` | String | `'1'` | Selects specific CFS ensemble member partitions (e.g., `'1'`). |

---

### Primary Forcing Product Mapping (`InputForcings`)

| ID | Product Description | ID | Product Description |
| :---: | :--- | :---: | :--- |
| **1** | NLDAS GRIB Retrospective | **15** | Alaska 3km Alaska Nest |
| **2** | NARR GRIB Retrospective | **16** | Hawaii 3km NAM Nest (Radiation Only) |
| **3** | GFS GRIB2 Global Production (Full Gaussian) | **17** | Puerto Rico 3km NAM Nest (Radiation Only) |
| **4** | NAM Nest GRIB2 CONUS Production | **18** | WRF-ARW GRIB2 Puerto Rico |
| **5** | HRRR GRIB2 CONUS Production | **19** | HRRR Alaska GRIB2 Production |
| **6** | RAP GRIB2 CONUS 13km Production | **20** | Alaska Analysis & Assimilation (AnA) |
| **7** | CFSv2 6-hourly GRIB2 Global Production | **21** | AORC Alaska |
| **8** | WRF-ARW GRIB2 Hawaii Nest | **22** | Alaska Extended Analysis & Assimilation |
| **9** | GFS GRIB2 Global Production (0.25° Lat/Lon) | **23** | ERA5-Interim |
| **10** | Custom NetCDF Hourly Forcing (Primary) | **24** | National Blended Model (NBM) |
| **11** | Custom NetCDF Hourly Forcing (Secondary) | **25** | National Digital Forecast Database (NDFD) |
| **12** | AORC CONUS (Local or AWS S3) | **26** | HRRR 15-Minute Sub-hourly Cycling |
| **13** | Hawaii 3km NAM Nest | **27** | NWM v3 Retrospective Forcings (AWS S3) |
| **14** | Puerto Rico 3km NAM Nest | | |

---

## 5. Forecast Cycles & Time-Handling

Configurations managing real-time forecast cycles, retro/AnA lookback periods, and override flags.

| Parameter | Type / Options | Default / Example | Description |
| :--- | :--- | :--- | :--- |
| `AnAFlag` | `0` = Off<br>`1` = On | `0` | Controls Analysis & Assimilation (AnA) mode. Changes bias correction behavior and offset calculations. |
| `LookBack` | Minutes | `-9999` | Lookback window duration (in minutes) from `RefcstBDateProc` for AnA mode. Set to `-9999` if unused. |
| `RefcstBDateProc` | Timestamp | `202210071400` | Simulation reference start date (`YYYYMMDDHHMM`). Acts as the end date when `AnAFlag: 1`. |
| `ForecastFrequency` | Minutes | `60` | Frequency (in minutes) at which forecast cycles are generated (`60` for hourly retrospective). |
| `ForecastShift` | Minutes | `0` | Minute offset to shift cycle origin times (e.g., shifting 00/06/12/18 UTC by 60 min to 01/07/13/19 UTC). |
| `qpf` | `0` = Off<br>`1` = On | `0` | Quantitative Precipitation Forecast flag. `1` zeroes out expected precipitation for the first 6 hours of a cycle. |
| `coldstart` | `0` = Off<br>`1` = On | `0` | Overrides `qpf` for forecast hour 1 (permitting rainfall), then zeroes precipitation for hours 2–6 when hotstart is unavailable. |

---

## 6. Downscaling & Bias Correction Options

Selectable topographic downscaling algorithms and climatological bias-correction routines.

### Downscaling Configurations

| Parameter | Type / Options | Default / Example | Description |
| :--- | :--- | :--- | :--- |
| `TemperatureDownscaling` | Integer Array | `[3, 3]` | `0` = None<br>`1` = Static lapse rate ($6.75^\circ\text{C/km}$)<br>`2` = Pre-calculated regridded lapse rate (NWM only)<br>`3` = Dynamic timestep lapse rate |
| `PressureDownscaling` | Integer Array | `[1, 1]` | `0` = None<br>`1` = Hypsometric scaling using terrain elevation delta |
| `ShortwaveDownscaling` | Integer Array | `[1, 1]` | `0` = None<br>`1` = Topographic solar slope/aspect adjustment |
| `PrecipDownscaling` | Integer Array | `[0, 0]` | `0` = None<br>`1` = Mountain Mapper algorithm using PRISM monthly climo |
| `HumidityDownscaling` | Integer Array | `[1, 1]` | `0` = None<br>`1` = Extrapolated specific humidity using downscaled T & P |
| `DownscalingParamDirs` | String Array | `["./forcingParam/AnA", "./forcingParam/AnA"]` | Directory paths containing downscaling grid parameters (legacy NWM WRF-Hydro domain only). |

---

### Bias Correction Configurations

| Parameter | Type / Options | Default / Example | Description |
| :--- | :--- | :--- | :--- |
| `TemperatureBiasCorrection` | Integer Array | `[0, 4]` | `0` = None<br>`1` = CFSv2-NLDAS2 Parametric<br>`2` = HRRRv3 Diurnal<br>`3` = GFS Parametric<br>`4` = HRRR Parametric |
| `PressureBiasCorrection` | Integer Array | `[0, 0]` | `0` = None<br>`1` = CFSv2-NLDAS2 Parametric Distribution |
| `HumidityBiasCorrection` | Integer Array | `[0, 0]` | `0` = None<br>`1` = CFSv2-NLDAS2 Parametric<br>`2` = HRRRv3 Diurnal |
| `WindBiasCorrection` | Integer Array | `[0, 4]` | `0` = None<br>`1` = CFSv2-NLDAS2 Parametric<br>`2` = HRRRv3 Diurnal<br>`3` = GFS Parametric<br>`4` = HRRR Parametric |
| `SwBiasCorrection` | Integer Array | `[0, 2]` | `0` = None<br>`1` = CFSv2-NLDAS2 Parametric<br>`2` = HRRRv3 Analysis Adjustment |
| `LwBiasCorrection` | Integer Array | `[0, 2]` | `0` = None<br>`1` = CFSv2-NLDAS2 Parametric<br>`2` = HRRRv3 Blanket Adjustment<br>`3` = GFS Parametric |
| `PrecipBiasCorrection` | Integer Array | `[0, 0]` | `0` = None<br>`1` = CFSv2-NLDAS2 Parametric Distribution |

---

## 7. Supplemental Precipitation Options

Layered precipitation product options used to override or supplement primary forcing precipitation fields.

### Parameters & Settings

| Parameter | Type / Options | Default / Example | Description |
| :--- | :--- | :--- | :--- |
| `SuppPcp` | Integer Array | `[1, 5, 13]` | Selects supplemental precipitation product IDs to layer into final forcing fields. |
| `SuppPcpForcingTypes` | String Array | `["GRIB2", "GRIB2", "GRIB2"]` | Native file format per product in `SuppPcp`. Valid options: `GRIB1`, `GRIB2`, `NETCDF`. |
| `SuppPcpDirectories` | String Array | `["./MRMS_GAUGE", "./MRMS_MULTI", "./MRMS_CLASS"]` | Directory paths to search for input supplemental precipitation files. |
| `SuppPcpParamDir` | String Array | `["./forcingParam/AnA", "./forcingParam/AnA", "./forcingParam/AnA"]` | Directory paths containing supplemental parameter fields (e.g., monthly RQI climatology; legacy NWM domain only). |
| `RegridOptSuppPcp` | Integer Array | `[1, 1, 1]` | ESMF regridding method per supplemental product: `1` = Bilinear, `2` = Nearest Neighbor, `3` = Conservative Bilinear. |
| `SuppPcpTemporalInterpolation` | Integer Array | `[0, 0, 0]` | Pre-downscaling/bias-correction interpolation: `0` = None, `1` = Nearest Neighbor, `2` = Linear Weighted. |
| `SuppPcpInputOffsets` | Integer Array | `[0, 0, 0]` | Offset (in hours) from available forecast cycle and 00z for AnA runs. |
| `SuppPcpMandatory` | Binary Array | `[0, 0, 0]` | `1` = Mandatory, `0` = Optional. Determines fallback behavior if supplemental precipitation files are missing. |
| `RqiMethod` | Integer | `2` | Radar Quality Index filtering method: `0` = Off (use all radar QPE), `1` = Hourly MRMS RQI grids, `2` = NWM monthly climatology RQI. |
| `RqiThreshold` | Float (`0.0`–`1.0`) | `0.9` | RQI cutoff threshold below which radar QPE cells are masked out (e.g., `0.9`). |

---

### Supplemental Product Mapping (`SuppPcp`)

| ID | Supplemental Product Description | ID | Supplemental Product Description |
| :---: | :--- | :---: | :--- |
| **1** | MRMS GRIB2 Hourly Radar-Only QPE | **9** | NBM Alaska Medium-Range |
| **2** | MRMS GRIB2 Hourly Gauge-Corrected Radar QPE | **10** | Alaska MRMS (No Liquid Fraction) |
| **3** | WRF-ARW 2.5km 48-hr Hawaii Nest Precip | **11** | Alaska Stage IV NWS Precip |
| **4** | WRF-ARW 2.5km 48-hr Puerto Rico Nest Precip | **12** | CONUS Stage IV NWS Precip |
| **5** | CONUS MRMS GRIB2 Hourly MultiSensor QPE | **13** | MRMS PrecipFlag Classification Product |
| **6** | Hawaii MRMS GRIB2 Hourly MultiSensor QPE | **14** | Custom Frequency Sub-hourly Precip |
| **7** | MRMS SBCv2 Liquid Water Fraction (NetCDF) | **15** | NBM Puerto Rico |
| **8** | NBM CONUS Medium-Range | | |

---

## 8. Download & Preprocessing Retry Settings

Optional properties controlling automated network downloads and file availability checks during dynamic preprocessing steps.

| Parameter | Type / Options | Default / Example | Description |
| :--- | :--- | :--- | :--- |
| `max_download_attempts` | Integer | `10` | Maximum number of retry attempts to download a given forcing file upon initial failure. |
| `download_attempt_interval` | Seconds | `10` | Waiting period (in seconds) between consecutive download retry attempts. |
| `check_file_availability` | `0` = Off<br>`1` = On | `0` | Toggles the preprocessing mechanism for checking remote forcing file availability prior to download. Defaults to `0`. |
