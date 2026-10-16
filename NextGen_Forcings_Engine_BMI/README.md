# NextGen Forcings Engine BMI Overview

The **NextGen Forcings Engine BMI** workflow is a meteorological forcing provider for the NextGen Water Resources Modeling Framework. It can be integrated directly as a supporting Basic Model Interface (BMI) forcing provider within a given NextGen formulation or executed as a standalone workflow to produce NextGen-compatible forcing files (NetCDF4 or catchment CSV files).

This application streamlines forcing file production for all National Water Model (NWM) v3.1 operational configurations across gridded domains, unstructured meshes, and the NextGen hydrofabric geopackage.

---

## Workflow Execution Pipeline

```text
[ config.yml ] 
      │
      ▼
[ ESMF Mesh Production (Target NextGen Hydrofabric / Domain) ]
      │
      ▼
[ Forcing File Download (Based on NWMv3.0 Operational Config) ]
      │
      ▼
[ ESMF Bilinear Regridding (Grid Cells, Nodes, Elements) ]
      │
      ├──► NextGen-Compatible Lumped Forcing Files (NetCDF or Catchment CSVs)
      └──► BMI Advertisement of Regridded Output Fields (via Catchment IDs)
```


## Component Directories

| Directory | Description |
| :--- | :--- |
| `Forcing_Extraction_Scripts` | Contains domain-specific scripts (CONUS, Alaska, Puerto Rico, Hawaii) to download required meteorological forcing data products from AWS S3 buckets or NOMADS servers for each regional NWM v3.1 setup. |
| `ESMF_Mesh_Domain_Configuration_Production` | Utility scripts focused on converting raw model domain file formats (NextGen Hydrofabric Geopackages, D-FlowFM, SCHISM) into ESMF-mesh-compliant NetCDF files. |
| `BMI_NextGen_Configs/` | Contains `config.yml` templates pre-configured for NWM v3.1 operational runs across gridded, coastal unstructured, and hydrofabric domains. |
| `NextGen_Domains/` | Contains sample domain geogrid files for gridded domains, coastal unstructured meshes, and the NextGen Hydrofabric (VPU 06). |
| `NWM_Params/` | Directory designated for supporting climatology files used in bias calibration and downscaling routines. |
| `NextGen_Forcings_Engine/` | Directory harboring the NextGen Forcings Engine BMI code (`bmi_model.py`), model implementation (`model.py`), and supporting Python tools for the Forcing Engine. |
| `NextGen_Forcings_Engine/core/` | Core Python supporting modules required by `model.py` to drive regridding, interpolation, and I/O. |

---

## Setup and Execution Guide

1. **Construct Required Python Environment**
   Navigate to the [Environment Setup](#environment-setup) within this README.md file and follow instructions accordingly. 

1. **Configure Execution Options**:
   Navigate to `./BMI_NextGen_Configs/CONUS/Ensemble_MVP_Config_Files` and select the appropriate template for your domain and NWM v3.1 configuration. Copy the selected `config.yml` into the working directory containing `bmi_wrapper.py`.

2. **Execute the BMI Workflow**:
   Run the driver script via `mpirun` specifying the path to your configuration file, output directory, and total MPI processes:

   ```bash
   mpirun -n 1 python bmi_wrapper.py ./short_range_config.yml -output_path ./Scratch/Short_Range -np 2

* **`./short_range_config.yml`**: Pathway to the target configuration file.
   * **`-output_path ./Scratch/Short_Range`**: Directory path for downloaded forcing inputs and regridded outputs.
   * **`-np 2`**: Number of internal MPI worker processes allocated to the Forcings Engine.

---

## Core System Architecture

### 1. Initialization Phase (`model.py`)
1. **Option Parsing (`config.py`)**: Reads the input configuration file, initializes `ConfigOptions`, and sets operational parameters, directory pathways, and operational flags.
2. **Execution Metadata**: Captures operational arguments (`nwm_version`, `nwm_config`) and sets logging pathways.
3. **Parallel Context (`parallel.py`)**: Sets up MPI communicator ranks and sizes (`MPI_COMM_WORLD`).
4. **Domain Processing (`geoMod.py`)**: Opens the modeling domain NetCDF/Geopackage, reads spatial coordinates and grid spacing, and broadcasts local processor grid boundaries and ESMF grid objects to worker ranks.
5. **Coordinate Reference Metadata**: Rank 0 extracts global CRS and geospatial metadata needed for NetCDF output generation.
6. **I/O Grid Allocation (`ioMod.py`)**: Allocates local slabs (`ny_local`, `nx_local`) on each rank to hold output array slices and initializes output NetCDF dataset structures.
7. **Input Class Allocation (`forcingInputMod.py`)**: Maps source data product characteristics (grid size, variables, projection) and creates ESMF input grid/mesh objects for downscaling and regridding.
8. **Supplemental Precip Setup (`suppPrecipMod.py`)**: Initializes supplemental precipitation dataset variables, disaggregation flags, and input ESMF regrid objects if enabled.
9. **BMI Execution Invocation**: Passes configured classes to `model.py` to drive time-stepping execution.

### 2. Operational Forecast Regridding Loop
1. **Supplemental Disaggregation Check (`disaggregateMod.py`)**: If supplemental 6-hourly accumulation data is supplied, disaggregates fields down to 1-hourly intervals prior to regridding.
2. **Forecast Cycle Configuration (`time_handling.py`)**: Computes current cycle numbers, creates output forecast directories, and configures lookback windows for Analysis and Assimilation (AnA) runs.
3. **Timestep Processing Loop**:
   * **Reset Grids**: Re-initializes destination arrays to missing values (`globalNdv`).
   * **Input Regridding (`forcingInputMod.py` / `regrid.py`)**: Identifies surrounding input cycle files, converts incoming GRIB2 files to NetCDF via `$WGRIB2`, and regrids forcing variables using predefined ESMF bilinear objects.
   * **Unit Conversion & Masking**: Converts average precipitation rates to instantaneous rates and sets unmapped out-of-domain cells to missing values.
   * **Post-Processing (`timeInterpMod.py` / `bias_correction.py` / `downscale.py`)**: Applies temporal interpolation to the target timestep, followed by bias correction and terrain downscaling routines where applicable.
   * **Supplemental Layering (`layeringMod.py` / `suppPrecipMod.py`)**: Regrids supplemental precipitation fields, applies temporal disaggregation/interpolation, and overlays (relayers) the result onto the output object.
4. **Finalization (`ioMod.py`)**: Closes dataset handles, writes output NetCDF/CSV files or advertises fields via BMI pointers, purges scratch directories, and terminates the MPI communicator.

---

## Key Workflow Files

* **`./NextGen_Forcings_Engine/bmi_model.py`**: The main BMI class definition exposing standard BMI control endpoints.
* **`./NextGen_Forcings_Engine/model.py`**: The core execution driver that controls data movement and transformation logic.
* **`bmi_wrapper.py`**: The primary entry-point script that automates ESMF mesh generation, forcing extraction, and execution.
* **`run_bmi_model.py`**: Lightweight script demonstrating standalone initialization, update, and finalization calls.
* **`run_bmi_unit_test.py`**: Comprehensive unit-test suite verifying BMI interface compliance.
* **`environment.yml`**: Conda environment definition containing all library dependencies (ESMF, NetCDF4, PyProj, Dask, mpi4py).

---

## Environment Setup

> [!IMPORTANT]  
> **HPC System Notice:** Building this environment on HPC systems requires linking Python bindings (`mpi4py`, `netCDF4`, `ESMF`) against the cluster’s system-installed MPI and NetCDF libraries. Simple Conda binary installs will often fail or create runtime ABI conflicts.

### Option 1: Standard Local Setup (Workstations / Single Nodes)
For local testing or non-HPC Linux environments using standard Conda binaries:

```bash
conda env create -f environment.yml
conda activate bmi_test
```

### Option 2: HPC System Setup (Module-Linked Build)

On HPC systems (e.g., Slurm/PBS clusters), load the system compiler, MPI, and NetCDF modules before creating the environment to ensure `mpi4py` and `ESMF` build against the host interconnect.

#### 1. Load System Modules
Load your cluster's designated toolchain (adjust module names based on your cluster environment):

```bash
module purge
module load gcc/11.x          # or intel/compiler
module load openmpi/4.x       # or mpich / mvapich2 / cray-mpich
module load netcdf-c/4.x
module load netcdf-fortran/4.x
```

#### 2. Create Base Conda Environment
Create a clean Conda environment containing the supporting Python packages (data manipulation, plotting, GIS, BMI interfaces) available via `conda-forge`. Crucially, we *exclude* `mpi4py`, `netCDF4`, `esmf`, and `esmpy` from this step, as they must be compiled against system libraries later.

```bash
# Create the base environment with core, non-compiled dependencies
conda create -n bmi_test -c conda-forge \
  python=3.10 \
  numpy \
  scipy \
  pandas \
  xarray \
  dask \
  zarr \
  s3fs \
  geopandas \
  shapely \
  pyproj \
  cfgrib \
  beautifulsoup4 \
  matplotlib \
  bokeh \
  pyyaml \
  ruamel.yaml \
  python-dotenv \
  bmipy

# Activate the new environment
conda activate bmi_test
```

#### 3. Compile `mpi4py` Against System MPI
Build `mpi4py` from source using the system's active `mpicc` wrapper:

```bash
MPICC=$(which mpicc) pip install --no-binary mpi4py mpi4py
```

#### 4. Compile & Link Parallel NetCDF4 (`netcdf4-python`)
Because the code base leverages parallel NetCDF I/O alongside MPI directives, `netCDF4` must be built from source against your HPC system's parallel-enabled NetCDF-C library (rather than using standard non-MPI Conda binaries):

```bash
# Point to your system's active NetCDF installation directory
export NETCDF4_DIR=$NETCDF_ROOT   # Or path from: nc-config --prefix

# Compile netCDF4 from source linking against system NetCDF + MPI
NETCDF4_DIR=$NETCDF4_DIR pip install --no-binary netcdf4 netcdf4
```

#### 5. Build & Link ESMF (ESMPy)
For proper parallel regridding, `esmpy` must be compiled or linked against the active system `ESMF` library build.

##### If ESMF is pre-configured on your system:
Load the module and install the Python bindings matching the system library version:

```bash
module load esmf
pip install esmpy --no-deps
```

##### If ESMF is NOT installed or configured:
If your cluster administrator has not provided a pre-built `esmf` module, you must install or compile the core C++/Fortran ESMF library first:

* **Build from Source (Recommended for HPC performance):** Follow the official [ESMF Building and Installing Guide](https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node9.html) to build the library directly against your cluster's native MPI implementation and compiler suite. Once built, export the library path prior to installing `esmpy`:
  ```bash
  export ESMFMKFILE=/path/to/installed/esmf/lib/esmf.mk
  pip install esmpy --no-deps
```
  
*Note: Verify your build by confirming `python -c "import mpi4py, esmpy; print('MPI and ESMF loaded successfully')"` executes without library linking errors inside an interactive compute node (`srun` / `qsub`).*


