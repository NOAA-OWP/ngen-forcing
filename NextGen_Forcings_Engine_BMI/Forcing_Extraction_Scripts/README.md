# Forcing Extraction Workflow

The `Forcing_Extraction_Scripts` subsystem provides modular downloaders for retrieving and staging atmospheric and hydrologic model forcing data required by the National Water Model v4 (NWMv4) NextGen framework.

---

## Architecture Overview

All downloaders inherit from `ForecastDownloader` (`forecast_download_base.py`), establishing a standard lifecycle across datasets:
1. **Cycle Window Evaluation**: Computes lookback/lagback windows aligned to model operational cadences, optionally limited to the configured forecast horizon.
2. **Cleanup**: Removes data of cycles older than the window (standalone runs only; never data the window needs).
3. **Download**: Fetches each needed file with retries and timeouts, verifies the byte count (and the MD5 when the ETag is a plain MD5), and publishes it atomically. Interrupted transfers resume with `Range`/`If-Range`.
4. **Availability Verification** (`check_file_availability: 1`): Sends `HTTP HEAD` requests instead of downloading and classifies each file as missing upstream, not downloaded, current, changed upstream, size mismatch, or without stored validators.

### Change detection

Each downloaded file gets a sidecar `.<name>.meta.json` recording the upstream `ETag`, `Last-Modified` and size. On later runs:

- **Products rewritten upstream** (`mutable_upstream`: Stage IV, whose analyses are re-issued about 24 h after valid time, and NDFD) are revalidated with `If-None-Match` / `If-Modified-Since`; a `304 Not Modified` keeps the local file, a `200` replaces it. NDFD stores validators per valid-period part.
- **Other products** are not requested again unless the local size disagrees with the recorded size. `--revalidate` forces conditional requests for them too.
- If an existing file cannot be revalidated (for example the server is down, or NDFD now holds an issuance from another cycle), the existing file is kept.
- A `404` for a cycle more than 48 h old is final and is not retried.

File modification times are set from `Last-Modified`.

`FixedFileDownloader` covers observation products (MRMS, Stage IV) with one or more fixed files per hour; `MRMSMultiSensorDownloader` covers the hourly MRMS MultiSensor products of every MRMS domain. NDFD additionally concatenates its valid-period bins into one GRIB2 file per field.

A failed download raises `DownloadError`; the standalone scripts and `forcing_extraction.py` exit with status 1.

---

## Registered Forecast Forcing Datasets (`AnAFlag: 0`)

Registered in `FORCING_SRC` in `NextGen_Forcings_Engine_BMI/forcing_extraction.py`. Downloads are limited to the hours needed for `ForecastInputHorizons`.

| Forcing ID | Domain | Model / Product | Script | Source Endpoint | Forecast Hours Available |
|:---:|:---|:---|:---|:---|:---:|
| **3** | Global | GFS | `Global/get_prod_GFS.py` | AWS S3 (`noaa-gfs-bdp-pds`) | 1–384 |
| **5** | CONUS | HRRR | `CONUS/get_conus_HRRR.py` | AWS S3 (`noaa-hrrr-bdp-pds`) | 0–48 (00/06/12/18Z), 0–18 otherwise |
| **6** | CONUS | RAP (`pgrb` and `bgrb`) | `CONUS/get_conus_RAP.py` | AWS S3 (`noaa-rap-pds`) | 0–51 (03/09/15/21Z), 0–18 otherwise |
| **7** | Global | CFSv2 (members 01–04) | `Global/get_CFSv2.py` | AWS S3 (`noaa-cfs-pds`) | 0–720, 6-hourly |
| **8** | Hawaii | HIRESW WRF-ARW | `Hawaii/get_ARW_Hawaii.py` | NOMADS HTTPS | 1–48 |
| **13** | Hawaii | NAM Nest Hawaii | `Hawaii/get_prod_NAM_Nest_Hawaii.py` | AWS S3 (`noaa-nam-pds`) | 1–60 |
| **14** | Puerto Rico | NAM Nest Puerto Rico | `Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico.py` | AWS S3 (`noaa-nam-pds`) | 1–60 |
| **15** | Alaska | NAM Nest Alaska | `Alaska/get_prod_NAM_Nest_Alaska.py` | AWS S3 (`noaa-nam-pds`) | 1–60 |
| **16** | Hawaii | NAM Nest Hawaii (radiation only) | `Hawaii/get_prod_NAM_Nest_Hawaii.py` | AWS S3 (`noaa-nam-pds`) | 1–60 |
| **17** | Puerto Rico | NAM Nest Puerto Rico (radiation only) | `Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico.py` | AWS S3 (`noaa-nam-pds`) | 1–60 |
| **18** | Puerto Rico | HIRESW WRF-ARW | `Puerto_Rico/get_ARW_Puerto_Rico.py` | NOMADS HTTPS | 1–48 |
| **19** | Alaska | Alaska HRRR | `Alaska/get_Alaska_HRRR.py` | AWS S3 (`noaa-hrrr-bdp-pds`) | 0–48 (00/06/12/18Z), 0–18 otherwise |
| **24** | CONUS | NBM CONUS | `CONUS/get_prod_NBM_Conus.py` | AWS S3 (`noaa-nbm-grib2-pds`) | 1–264 |
| **25** | CONUS | NDFD CONUS | `CONUS/get_conus_NDFD.py` | AWS S3 (`noaa-ndfd-pds`, latest issuance only) | Day 1–7 (QPF Day 1–3) |
| **26** | CONUS | HRRR 15-minute | `CONUS/get_CONUS_HRRR_subhourly.py` | AWS S3 (`noaa-hrrr-bdp-pds`) | 0–18 |
| **supp8** | CONUS | NBM CONUS APCP | `CONUS/get_prod_NBM_Conus.py` | AWS S3 (`noaa-nbm-grib2-pds`) | 1–264 |
| **supp9** | Alaska | NBM Alaska APCP | `Alaska/get_prod_NBM_Alaska.py` | AWS S3 (`noaa-nbm-grib2-pds`) | 1–264 |
| **supp11** | Alaska | Stage IV + MRMS | `Alaska/get_Alaska_StageIV.py` (+ MRMS, see below) | NOMADS HTTPS | 6-hourly analyses |
| **supp12** | CONUS | Stage IV + MRMS | `CONUS/get_conus_StageIV.py` (+ MRMS, see below) | NOMADS HTTPS | hourly analyses |
| **supp15** | Puerto Rico | NBM Puerto Rico APCP | `Puerto_Rico/get_prod_NBM_Puerto_Rico.py` | AWS S3 (`noaa-nbm-grib2-pds`) | 1–264 |

> **Retired products**: NAM (forcings 13-17 and AnA 13/14) and HiResW Hawaii/Puerto Rico (8, 18) were retired on 2026-11-03 and replaced by RRFS ([SCN 26-47](https://www.weather.gov/media/notification/pdf_2026/SCN26-47_Updated_Retire_NAM_SREF_HREF_HiresW_NAM_MOS.aab.pdf), [SCN 26-48](https://www.weather.gov/media/notification/pdf_2026/scn26-048_Updated_RRFS_and_REFS_Implementation_aae.pdf)). Their downloaders still fetch cycles up to the last one (2026-11-03 06Z), for retrospective runs, and fail immediately for later cycles. The S3 NAM archive (`noaa-nam-pds`) holds the Alaska, Hawaii and Puerto Rico nests; HiResW Hawaii/Puerto Rico was only on NOMADS, which keeps about two days, so its history is not available from this source.

---

## Analysis & Assimilation Sources (`AnAFlag: 1`)

Registered in `FORCING_ANA_SRC`:

| Forcing ID | Domain | Product | Script |
|:---:|:---|:---|:---|
| **5** | CONUS | HRRR f01 | `CONUS/get_conus_HRRR_AnA.py` |
| **6** | CONUS | RAP f01 (`pgrb` and `bgrb`) | `CONUS/get_conus_RAP_AnA.py` |
| **13** | Hawaii | NAM Nest Hawaii f01–f06 of the covering 6-hourly cycle | `Hawaii/get_prod_NAM_Nest_Hawaii_AnA.py` |
| **14** | Puerto Rico | NAM Nest Puerto Rico f01–f06 of the covering 6-hourly cycle | `Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico_AnA.py` |
| **19** | Alaska | Alaska HRRR | `Alaska/get_Alaska_HRRR.py` |
| **20** | Alaska | Alaska HRRR f01–f04 | `Alaska/get_Alaska_HRRR_AnA.py` |
| **26** | CONUS | HRRR 15-minute f01 | `CONUS/get_conus_HRRR_subhourly_AnA.py` |
| **supp1** | CONUS | MRMS Radar-Only QPE | `CONUS/get_conus_MRMS_Radar.py` |
| **supp2** | CONUS | MRMS MultiSensor QPE (hourly) | `CONUS/get_conus_MRMS_MultiSensor.py` |
| **supp6** | Hawaii | MRMS MultiSensor QPE (hourly) | `Hawaii/get_MRMS_MultiSensor_Hawaii.py` |
| **supp10** | Alaska | MRMS MultiSensor QPE (hourly) | `Alaska/get_MRMS_MultiSensor_Alaska.py` |
| **supp11** | Alaska | Stage IV + MRMS | `Alaska/get_Alaska_StageIV.py` (+ MRMS, see below) |
| **supp12** | CONUS | Stage IV + MRMS | `CONUS/get_conus_StageIV.py` (+ MRMS, see below) |
| **supp15** | Puerto Rico | NBM Puerto Rico f001 | `Puerto_Rico/get_prod_NBM_Puerto_Rico_AnA.py` |
| **supp16** | CONUS | MRMS 2-minute precipitation rate | `CONUS/get_conus_MRMS_subhourly.py` |

Identifiers without an entry (for example 22, `supp5`) are skipped with a warning.

> **`supp11` / `supp12`**: the engine reads these as one `"<Stage IV dir>,<MRMS dir>"` input, so they must be the only `SuppPcp` option. Stage IV is downloaded into the first `SuppPcpDirectories` entry and MRMS MultiSensor (Alaska or CONUS) into the second, under `MultiSensor_QPE_01H_PassN/YYYYMMDD/`.
>
> **`supp16`**: sub-hourly MRMS is downloaded under `<dir>/MRMS_SubHourly/YYYYMMDD/`, and hourly MultiSensor MRMS for bias correction under `<dir>/MRMS_Hourly/MultiSensor_QPE_01H_PassN_00.00/YYYYMMDD/`.
>
> **MRMS MultiSensor** for `supp2`, `supp6` and `supp10` is stored flat under `MultiSensor_QPE_01H_PassN_00.00/`.

---

## Configuration & Pipeline Integration

Download parameters are set in the YAML configs in `BMI_NextGen_Configs/`:

```yaml
# Network resilience parameters
max_download_attempts: 10
download_attempt_interval: 10
check_file_availability: 0

# Forcing specification
InputForcings: [25]                 # CONUS NDFD
InputForcingDirectories: ["./data"] # NDFD files go to ./data/NDFD/YYYYMMDD/wgrbbul/ndfd_conus/
ForecastInputHorizons: [10080]      # minutes (7 days)

# Supplemental precipitation parameters
SuppPcp: []
SuppPcpDirectories: ''
```

## Example to Execute the Workflow

Follow these steps from the root of the repository to set up the minimal environment and execute the extraction workflow end-to-end:

```bash
# 1. Navigate to the Forcing Extraction Script Directory
cd ngen-forcing/NextGen_Forcings_Engine_BMI/Forcing_Extraction_Scripts

# 2. Create the dedicated conda environment from the minimal specification
conda env create -f environment.yml

# 3. Activate the environment
conda activate forcing_extraction

# 4. Navigate into the BMI engine directory
cd ../

# 5. Execute the extraction workflow with the target hydrofabric configuration
python forcing_extraction.py ./BMI_NextGen_Configs/CONUS/Medium_Range_Blend/hydrofabric/config.yml
```

Individual downloaders can also be run directly as modules from `NextGen_Forcings_Engine_BMI/`:

```bash
python -m Forcing_Extraction_Scripts.CONUS.get_conus_HRRR ./HRRR "2026-11-10 12:00:00" --lookBackHours 6
```

## Testing

From `NextGen_Forcings_Engine_BMI/Forcing_Extraction_Scripts` (requires `pytest`):

```bash
python -m pytest tests            # offline tests; upstream servers are simulated
python -m pytest tests --live     # also probe every downloader against its live upstream server
```

The live probe can also be run on its own from `NextGen_Forcings_Engine_BMI/`. For each downloader it picks a recently published cycle, lists the files the downloader would fetch, and checks a sample of them with `HEAD` plus a ranged `GET` that must contain a GRIB2 message:

```bash
python -m Forcing_Extraction_Scripts.probe_sources                      # all scripts, three files each
python -m Forcing_Extraction_Scripts.probe_sources CONUS/get_conus_HRRR.py --all
```

`check_server_behavior` re-checks the assumptions the downloaders make about server metadata: that HEAD and GET agree, that `Content-Length` matches the bytes served, that a plain 32-hex ETag is the MD5 of the content, that no transfer compression is applied, and that the content is GRIB2 as named. It downloads a sample of each product in full (up to `--max-mb`, default 64), and also notes whether each server would compress transfers if allowed to (the downloaders request identity encoding):

```bash
python -m Forcing_Extraction_Scripts.check_server_behavior                       # all scripts
python -m Forcing_Extraction_Scripts.check_server_behavior Global/get_prod_GFS.py --max-mb 400
python -m Forcing_Extraction_Scripts.check_server_behavior --url https://host/path/file.grib2
```

The downloaders report the same kinds of disagreement at run time: a body whose length differs from `Content-Length`, or whose content is not what its name says, is logged at ERROR as a server inconsistency and retried.
