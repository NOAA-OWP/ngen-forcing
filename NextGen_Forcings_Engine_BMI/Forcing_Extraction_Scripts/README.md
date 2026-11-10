# Forcing Extraction Workflow

The `Forcing_Extraction_Scripts` subsystem provides modular downloaders for retrieving and staging atmospheric and hydrologic model forcing data required by the National Water Model v4 (NWMv4) NextGen framework.

---

## Architecture Overview

All downloaders inherit from `ForecastDownloader` (`forecast_download_base.py`), establishing a standard lifecycle across datasets:
1. **Cycle Window Evaluation**: Computes lookback/lagback windows aligned to model operational cadences, optionally limited to the configured forecast horizon.
2. **Cleanup**: Removes data of cycles older than the window (standalone runs only; never data the window needs).
3. **Download**: Fetches each missing file with retries and timeouts, verifies the byte count, and publishes it atomically.
4. **Availability Verification** (`check_file_availability: 1`): Sends `HTTP HEAD` requests instead of downloading and reports which files are missing upstream.

`FixedFileDownloader` covers observation products (MRMS, Stage IV) with one or more fixed files per hour; `MRMSMultiSensorDownloader` covers the hourly MRMS MultiSensor products of every MRMS domain. NDFD additionally concatenates its valid-period bins into one GRIB2 file per field.

A failed download raises `DownloadError`; the standalone scripts and `forcing_extraction.py` exit with status 1.

---

## Registered Forecast Forcing Datasets

Each standard forecast pipeline is indexed by an operational ID registered in `NextGen_Forcings_Engine_BMI/forcing_extraction.py`:

| Forcing ID | Domain | Model / Product | Downloader Class / Script | Source Endpoint | Supported Forecast Horizon |
|:---:|:---|:---|:---|:---|:---:|
| **1** | CONUS | HRRR | `CONUS/get_HRRR.py` | AWS NODD / NOMADS | 1–48 Hours |
| **2** | CONUS | RAP | `CONUS/get_RAP.py` | AWS NODD / NOMADS | 1–21 Hours |
| **3** | CONUS | NAM Nest | `CONUS/get_NAM_Nest.py` | NOMADS / AWS S3 | 1–60 Hours |
| **4** | Global | GFS | `Global/get_GFS.py` | NOMADS / AWS S3 | 1–384 Hours |
| **7** | Global | CFSv2 | `Global/get_CFSv2.py` | NOMADS | Seasonal / Subseasonal |
| **8** | Hawaii | HIRESW WRF-ARW | `Hawaii/get_AWR_Hawaii.py` | NOAA NOMADS HTTPS | 1–48 Hours |
| **13** | Hawaii | NAM Nest Hawaii | `Hawaii/get_prod_NAM_Nest_Hawaii.py` | NOMADS / AWS S3 | 1–60 Hours |
| **14** | Puerto Rico | NAM Nest PR | `Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico.py` | NOMADS / AWS S3 | 1–60 Hours |
| **16** | Hawaii | NAM Nest Hawaii | `Hawaii/get_prod_NAM_Nest_Hawaii.py` | NOMADS / AWS S3 | 1–60 Hours |
| **17** | Puerto Rico | NAM Nest PR | `Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico.py` | NOMADS / AWS S3 | 1–60 Hours |
| **18** | Puerto Rico | HIRESW WRF-ARW | `Puerto_Rico/get_ARW_Puerto_Rico.py` | NOAA NOMADS HTTPS | 1–48 Hours |
| **19** | Alaska | Alaska HRRR | `Alaska/get_Alaska_HRRR.py` | AWS NODD / NOMADS | 1–48 Hours |
| **24** | CONUS | NBM CONUS | `CONUS/get_prod_NBM_Conus.py` | AWS NODD / NOMADS | 1–264 Hours |
| **25** | CONUS | NDFD CONUS | `CONUS/get_conus_NDFD.py` | NOAA NDFD AWS S3 PDS | Up to 168 Hours (Day 7) |

---

## Analysis & Assimilation (AnA) and Supplemental Precipitation Sources

When running in Analysis and Assimilation mode (`ana_flag: 1`) or incorporating supplemental precipitation grids (`SuppPcp`), the engine dispatches downloads via `forcing_ana_src`:

| Key / Identifier | Domain | Description / Product | Extraction Script |
|:---:|:---|:---|:---|
| **5** | CONUS | HRRR Analysis and Assimilation | `CONUS/get_conus_HRRR_AnA.py` |
| **6** | CONUS | RAP Analysis and Assimilation | `CONUS/get_conus_RAP_AnA.py` |
| **13** | Hawaii | NAM Nest Hawaii | `Hawaii/get_prod_NAM_Nest_Hawaii.py` |
| **14** | Puerto Rico | NAM Nest Puerto Rico AnA | `Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico_AnA.py` |
| **19** | Alaska | Alaska HRRR | `Alaska/get_Alaska_HRRR.py` |
| **20** | Alaska | Alaska HRRR Analysis and Assimilation | `Alaska/get_Alaska_HRRR_AnA.py` |
| **26** | CONUS | HRRR Subhourly Analysis and Assimilation | `CONUS/get_conus_HRRR_subhourly_AnA.py` |
| **supp1** | CONUS | MRMS Radar-Only QPE | `CONUS/get_conus_MRMS_Radar.py` |
| **supp2** | CONUS | MRMS Multi-Sensor QPE (Hourly) | `CONUS/get_conus_MRMS_MultiSensor.py` |
| **supp6** | Hawaii | MRMS Multi-Sensor Hawaii | `Hawaii/get_MRMS_MultiSensor_Hawaii.py` |
| **supp10** | Alaska | MRMS Multi-Sensor Alaska | `Alaska/get_MRMS_MultiSensor_Alaska.py` |
| **supp11** | Alaska | Stage IV Precipitation Alaska | `Alaska/get_Alaska_StageIV.py` |
| **supp12** | CONUS | Stage IV Precipitation CONUS | `CONUS/get_conus_StageIV.py` |
| **supp15** | Puerto Rico | NBM Puerto Rico Analysis and Assimilation | `Puerto_Rico/get_prod_NBM_Puerto_Rico_AnA.py` |
| **supp16** | CONUS | MRMS Multi-Sensor Subhourly QPE | `CONUS/get_conus_MRMS_subhourly.py` |

> **Note on `supp16`**: When `supp16` is designated, the extraction workflow dynamically binds and fetches both subhourly (`supp16`) and hourly (`supp2`) MRMS files under respective `/MRMS_SubHourly` and `/MRMS_Hourly` subdirectories.

---

## Configuration & Pipeline Integration

Download parameters are set across YAML configs in `BMI_NextGen_Configs/`:

```yaml
# Network resilience parameters
max_download_attempts: 10
download_attempt_interval: 10
check_file_availability: 0

# Forcing specification
Forcing:
  - 25  # CONUS NDFD
ForcingDirectories:
  - "./NDFD"
ForecastInputHorizons:
  - 168

# Supplemental precipitation parameters
SuppPcp: []
SuppPcpDirectories: ""
```

## Example to Execute the Workflow

Follow these steps from the root of the repository to set up the minimal environment and execute the extraction workflow end-to-end:

```bash
# 1. Navigate to the Forcing Extraction Script Directory
cd ngen-forcing/NextGen_Forcings_Engine_BMI/Forcing_Extraction_Scripts

# 2. Create the dedicated conda environment from the minimal specification
conda env create -f environment.yml

# 2. Activate the environment
conda activate forcing_extraction

# 3. Navigate into the BMI engine directory
cd ../

# 4. Execute the extraction workflow with the target hydrofabric configuration
python forcing_extraction.py ./BMI_NextGen_Configs/CONUS/Medium_Range_Blend/hydrofabric/config.yml
