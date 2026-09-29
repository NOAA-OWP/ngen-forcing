# NextGen Forcings Engine Repository Overview
Welcome to the NextGen Forcings Engine GitHub repository. This repository currently contains Python tools that allows the NextGen Water Resources Modeling Framework to be provide meteorological forcings data to NextGen formulations through either (1) csv catchment/netcdf files or (2) a Basic Model Interface (BMI) Forcings Engine. This repository also contains Forcings Extraction script repository that will allow a given user to extract all meteorological forcing data products required for National Water Model (NWM) version 3.0 operational configurations. ~~Future endeavors for this repository will also evenetually provide users more Python tools that will implement various data assimilation techniques (dyanmic data, oceanic/freshwater circuluation models) for NextGen formulations within the framework. ~~

**Documentation status:** Current development integrates the BMI Forcings Engine into the NextGen/NWM v4 stack. Some lower-level README content, example configurations, and terminology inherited from the original NWM v3 codebase may be outdated. See [DOCKER.md](DOCKER.md) for the current container build and test workflow.

# NextGen Lumped Forcings Driver Directory
This directory contains Python modules and a driver script that will provide users lumped meteorological forcings for catchments within the NextGen hydrofabric. Users will be able to create NextGen formatted csv catchment files or a single netcdf file that can be ingested by the default NextGen Forcings Provider. Current Python modules can support Analaysis of Record and Calibration (AORC) data, GFS data, CFS data, and HRRR data products that are needed for standard NWM operational configurations (Reanlysis, Analysis and Assimilation, Short range, Medium range, and long range). Setup, installation, and examples of utilizing these Python tools are further described within The ReadMe.md file in the directory as well as it's own Wiki Pages subsection. 

**Note:** The legacy Lumped Forcings Driver Docker image is not part of the current dependency-to-BMI-to-ngen image chain.

# NextGen Forcings Engine BMI Directory
This directory contains a BMI application that essentially streamlines the WRF-Hydro Forcings Engine into a BMI compliant data pipeline with universal regridding capabilites. ~~This Python BMI tool can directly provide the NextGen model engine regridded meteorological forcings that are required for all NWMv3.0 operational configurations. BMI realization configuration files have already been constructed for all NWMv3.0 operational configurations to support gridded domains, unstructured meshes, and the NextGen hydrofabric.~~ Setup, installation, and examples of utilizing these Python tools are further described within The ReadMe.md file in the directory as well as it's own Wiki Pages subsection. 

**Current status:** The configuration templates at [NextGen_Forcings_Engine_BMI/BMI_NextGen_Configs/config_templates/](NextGen_Forcings_Engine_BMI/BMI_NextGen_Configs/config_templates/) are the ones consumed by the NextGen/NWM v4 stack (by [`nwm-msw-mgr`](https://github.com/NGWPC/nwm-msw-mgr)). The older regional example configuration files were last updated on September 4, 2024. They might be supported by other workflows, but have not been tested in the NextGen/NWM v4 stack. The pytest suite exercises selected hydrofabric configurations. Current coastal workflows use the gridded discretization. The unstructured discretization has not been run with the latest NextGen/NWM v4 stack.

# Forcing Extraction Scripts Directory
This directory contains a series of scripts for each NWM domain subdirectory (CONUS, Alaska, Puerto Rico, Hawaii) that encompasses the required meteorological forcing data products needed for each regional NWMv3.0 operational configuration setup. Each script is a particular meteorlogical forcing data product that is available to download off the NOMADS server. Availability of each meteorlogical forcing data product varies, but a user can generally extract at least the last 24 hours of previous data products or forecast cycles available. Setup, installation, and examples of utilizing these Python tools are further described within The ReadMe.md file in the directory as well as it's own Wiki Pages subsection. 

**NWM version note:** These scripts had been organized around products used by NWM v3 configurations.  Their presence does not imply that every script or product is current for the NWM v4 stack.

# ESMF Mesh Domain Configuration Production Directory
This directory contains Python scripts that are only focused on coverting model domain file formats into a ESMF mesh compliant netcdf file that can be directly utilized by the NextGen Forcings Engine BMI. So far, this repository contains scripts to convert a NextGen hydrofabric geopackage or coastal model mesh file inputs (D-FlowFM, SCHISM) into ESMF mesh compliant netcdf files. Future updates to this repository will reflect more NextGen model formulations as they become available

# Streamflow Scripts Directory

~~This directory contains Python scripts for downloading and time-slicing real-time streamflow data from the USGS, US Army Corps of Engineers and Environment Canada. The time-slicing scripts process the native files into NetCDF files that can be directly used by T-Route for streamflow data assimilation. Setup, installation, and examples of utilizing these Python tools are further described within The README.md file in the directory.~~

**Note:** This was moved to the [`nwm-data-assimilation`](https://github.com/NGWPC/nwm-data-assimilation) repository. Outdated content above is retained in strikethrough.

# Observed Waterlevel Scripts Directory

~~This directory contains Python scripts for downloading and observed water level data from the NOAA CO-OPS. Setup, installation, and examples of utilizing these Python tools are further described within The README.md file in the directory.~~

**Note:** This was moved to the [`nwm-coastal`](https://github.com/NGWPC/nwm-coastal) repository. Outdated content above is retained in strikethrough.

# Coastal Forcing Scripts Directory

~~This directory contains Python scripts for creating forcing input files for coastal models. Meteorological forcings and tidal water levels input files for SFINCS and SCHISM can be created by these scripts. Setup, installation, and examples of utilizing these Python tools are further described within The README.md file in the directory.~~

**Note:** This was moved to the [`nwm-coastal`](https://github.com/NGWPC/nwm-coastal) repository. See that repository for the current coastal gridded forcing workflow. Outdated content above is retained in strikethrough.

# Tests Directory

`tests/` contains the golden-file tests that were added in 2026, covering certain classes for some forcing engine configurations, for the hydrofabric discretization type. See [tests/README.md](tests/README.md) for details.
