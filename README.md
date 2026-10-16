# NextGen Forcings Engine

Welcome to the **NextGen Forcings Engine** repository. This repository provides Python-based workflows that allow the NextGen Water Resources Modeling Framework to supply meteorological forcing data to NextGen formulations.

The engine supports two primary data delivery pipelines:
1. **Lumped Catchment / NetCDF Files**: CSV catchment files or NetCDF files produced via exact rasterization.
2. **Basic Model Interface (BMI)**: A BMI-compliant, real-time data pipeline for direct ingestion into NextGen formulations.

---

## Directory Overview

| Directory | Workflow Type | Key Capabilities |
| :--- | :--- | :--- |
| `NextGen Lumped Forcings Driver` | Standalone Driver Script & Modules | Generates catchment-lumped forcing CSVs or single NetCDF files using the `ExactExtract` rasterization tool. |
| `NextGen Forcings Engine BMI` | BMI Application Pipeline | Wraps and streamlines the [NCAR WRF-Hydro Forcings Engine](https://github.com/NCAR/WrfHydroForcing) into a BMI data pipeline with ESMF regridding. |

---

## 1. NextGen Lumped Forcings Driver Directory

This directory contains Python modules and a driver script designed to extract and lump meteorological forcings across catchments within the NextGen hydrofabric using the `ExactExtract` rasterization method. 

### Key Features
* **Output Formats**: Generates NextGen-compatible CSV catchment files or a consolidated NetCDF file for ingestion by the default NextGen Forcings Provider.
* **Supported Data Products**:
  * **AORC**: Analysis of Record and Calibration (local disk or AWS S3 bucket)
  * **GFS**: Global Forecast System
  * **CFS**: Climate Forecast System
  * **HRRR**: High-Resolution Rapid Refresh
* **Configuration Support**: Supports standard National Water Model (NWM) operational configurations, including Reanalysis, Analysis & Assimilation (AnA), Short-Range, Medium-Range, and Long-Range.

> **Documentation & Setup**: Setup, installation, and execution examples are further detailed in the `README.md` file within the directory.

---

## 2. NextGen Forcings Engine BMI Directory

This directory houses a Python-based Basic Model Interface (BMI) wrapper that converts the legacy [WRF-Hydro Forcings Engine](https://github.com/NCAR/WrfHydroForcing) into a streamlined, BMI-compliant data pipeline.

### Key Features
* **Universal Regridding**: Utilizes the Earth System Modeling Framework (ESMF) regridding suite to dynamically map forcing fields directly to NextGen domains.
* **Operational Capabilities**: Directly provides regridded forcing data required for all **NWM v3.1** operational configurations.
* **Pre-configured Realizations**: Pre-built BMI realization configuration files are included for:
  * Gridded domains
  * Unstructured meshes
  * NextGen hydrofabric
* **Integrated Preprocessing**: Automatically handles forcing file extraction and ESMF mesh generation directly as a preprocessing workflow before Forcing Engine execution

> **Documentation & Setup**: Setup, installation, and execution examples are further detailed in the `README.md` file within the directory.
