# NextGen Lumped Forcings Driver Overview

The **NextGen Lumped Forcings Driver** provides a Python workflow to extract, lump, and rasterize meteorological forcing data for catchments within the NextGen hydrofabric using the [`exactextract`](https://github.com/isciences/exactextract) tool. 

Users can produce NextGen-compatible catchment CSV files or consolidated NetCDF files derived from various operational meteorological products, including AORC, GFS, CFS, and HRRR.

---

## Installation & Setup

### 1. Create the Conda Environment
Build and activate the environment using the provided `environment.yml` file:

```bash
conda env create --name NextGen_Lumped_Forcings_Driver --file=environment.yml
conda activate NextGen_Lumped_Forcings_Driver
```
### 2. Install `exactextract` Python Bindings
The driver requires custom `exactextract` Python bindings for coverage fraction operations. Install them by following the instructions in the target repository branch:

* **Repository**: [GitHub - exactextract (coverage-fraction-pybindings branch)](https://github.com/jdalrym2/exactextract/tree/coverage-fraction-pybindings)


---

## Usage & Execution

### Command-Line Execution
You can directly execute the driver script from the terminal using `run.py`:

```bash
python run.py /pathway/to/hydrofabric/gpkg -o /pathway/to/forcing/output/directory
```

* **`hydrofabric`** *(Positional)*: File pathway to the NextGen hydrofabric GeoPackage (`.gpkg`) used to extract forcing data.
* **`-o / --output`** *(Flag)*: Pathway to the output directory where generated NetCDF/CSV NextGen forcing files will be stored.

---

### Python Module Usage Example

```python
from NextGen_lumped_forcings_driver import NextGen_lumped_forcings_driver

NextGen_lumped_forcings_driver(
    output_root="/pathway/to/lumped_forcings/output_directory",
    start_time="2010-01-01 00:00:00",
    end_time="2010-02-01 00:00:00",
    met_dataset="AORC",
    hyfabfile="/pathway/to/NextGen_hydrofabric_geopackage.gpkg",
    hyfabfile_parquet=None,
    met_dataset_pathway="/pathway/to/data/source",
    weights_file=None,
    netcdf=False,
    csv=True,
    bias_calibration=False,
    downscaling=False,
    CONUS=False,
    AnA=False,
    num_processes=1
)
```

## Function Parameter Reference

| Parameter | Type | Required For | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `output_root` | String | **All** | *None* | Root output directory for generated NetCDF/CSV files and temporary scratch data. |
| `met_dataset` | String | **All** | *None* | Target dataset. Options: `'AORC'`, `'GFS'`, `'HRRR'`, `'CFS'`. |
| `hyfabfile` | String | **All** | *None* | File path to the target NextGen hydrofabric GeoPackage (`.gpkg`). |
| `start_time` | String | AORC, HRRR | `None` | Start timestamp (`'YYYY-MM-DD HH:00:00'`) for reanalysis or AnA runs. |
| `end_time` | String | AORC | `None` | End timestamp (`'YYYY-MM-DD HH:00:00'`) for reanalysis data processing. |
| `met_dataset_pathway` | String | **All** | `None` | Path to source NetCDF/GRIB2 files or directory. For AORC, use `"s3://noaa-nws-aorc-v1-1-1km/"` for cloud extraction. |
| `hyfabfile_parquet` | String | Optional | `None` | Path to hydrofabric VPU/CONUS Parquet file containing terrain metadata for bias calibration/downscaling. |
| `weights_file` | String | Optional | `None` | Path to a pre-computed `exactextract` coverage fraction weights CSV to bypass re-computation. |
| `netcdf` | Boolean | Optional | `True` | Generates a consolidated NextGen NetCDF output file when `True`. |
| `csv` | Boolean | Optional | `False` | Generates individual NextGen catchment CSV files when `True`. |
| `bias_calibration` | Boolean | GFS, CFS, HRRR | `False` | Applies NCAR NWM Forcings Engine bias calibration routines. |
| `downscaling` | Boolean | GFS, CFS, HRRR | `False` | Applies NCAR NWM Forcings Engine terrain downscaling routines. |
| `CONUS` | Boolean | AORC | `False` | Flags whether the hydrofabric encompasses the entire CONUS network versus a local subset. |
| `AnA` | Boolean | HRRR | `False` | Enables Analysis & Assimilation mode for HRRR, producing a 28-hour lookback window (`f01`). |
| `num_processes` | Integer | Optional | `1` | Number of parallel worker threads to allocate for multi-threaded rasterization. |

---

## Important Execution Notes

> **Memory Guardrails for CONUS Runs**: Setting `CONUS=True` with `csv=True` is disabled because generating over 800,000 CSV files simultaneously will exhaust system I/O memory. Always use `netcdf=True` for full CONUS processing.

> **AORC AWS S3 Performance**: For long-term AORC reanalysis extractions, setting `met_dataset_pathway="s3://noaa-nws-aorc-v1-1-1km/"` alongside `num_processes > 1` significantly speeds up execution. Ensure your environment has sufficient RAM (>15–20 GB) allocated when loading full CONUS hydrofabric files.
