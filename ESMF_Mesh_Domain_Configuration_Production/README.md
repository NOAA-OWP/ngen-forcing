# 2026 Update

**WARNING:** As of September 2026, many aspects of this file may be outdated relative to the latest NextGen/NWM v4 stack, but this file has been kept mostly intact for posterity.

See the top-level repository [README.md](../README.md), [INSTALL.md](../INSTALL.md), and [DOCKER.md](../DOCKER.md) for the latest information.

# Overview
This subdirectory with the NextGen Forcings Github repository contains Python scripts that are only focused on converting model domain file formats into a ESMF mesh compliant netcdf file that can be directly utilized by the NextGen Forcings Engine BMI. So far, this repository contains scripts to convert a NextGen hydrofabric geopackage or coastal model mesh file inputs (D-FlowFM, SCHISM) into ESMF mesh compliant netcdf files. Future updates to this repository will reflect more NextGen model formulations as they become available 

# Setting Up Required Python Environment to Execute Forcing Extraction Scripts Using Anaconda
conda env create --name ngen_esmf_mesh_prod --file=environment.yml

# Ongoing Evaluations
~~The NextGen CONUS hydrofabric versions earlier than version 2.2 has issues with the floating point precision of the element centroids for the WGS85 coordinate reference system that casuses duplicate element centroids to be produced during the ESMF Mesh file production. They are current being addressed in evaluated with the upcoming NextGen hydrofabric v2.2 release and will ensure that the NextGen CONUS hydrofabric ESMF mesh file can be properly produced. For earlier NextGen hydrofabric versions, we advise to only construct ESMF mesh netcdf files for hydrofabric VPUs only or smaller subsets.~~

**2026 Update:** This evaluation was last updated on September 9, 2024 and has not been revalidated against current hydrofabric releases.  The hydrofabric version 2.2 mentioned is not supported by the NextGen/NWM v4 stack.
