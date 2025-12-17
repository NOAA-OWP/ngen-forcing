# Overview
This directory contains Python scripts to process FVCOM data for the Great Lake domains. The resulting files are in the format that can be used by SCHISM as the tidal time history at the boundary nodes. An example SCHISM tidal file is elev2D.th.nc. The description of the FVCOM data can be found at https://www.fvcom.org/, and the products from NOAA's CO-OPS for each of the Great Lake regions can be found at https://tidesandcurrents.noaa.gov/models.html. The SCHISM tidal boundary condition file format is documented at https://schism-dev.github.io/schism/master/input-output/optional-inputs.html.


# Prerequisite
In addition to the FVCOM products of the Great Lake domains, the SCHISM parameter file, open_bnds_hgrid.nc, is also needed as one of the input files.   


# Script description
makeOcean_fvcom.py : The main driver of the Python scripts.
TidesCurrentsProduct.py : The ABC of the NOAA's Operational Forecast System (OFS)
FVCOM.py : A subclass of TidesCurrentsProduct. Manages the Great Lakes FVCOM product from NOAA's Center for Operational Oceanographic Products and Services (CO-OPS) 
FVCOMCollection.py : Manages a collection of FVCOM product files for a given period and domain.
SCHISMGrid.py : The class for a SCHISM horizontal grid, hgrid.gr3,  file.
SCHISMOceanMaker.py : This class contains utilities to create a SCHISM elev2D.th.cn file


# Script Usage
Each script has a help option (-h) for printing usage information.

usage: makeOcean_fvcom.py [-h] fvcom_dir fvcom_domain schism_grid start_time end_time output

positional arguments:
  fvcom_dir     FVCOM inputfile directory
  fvcom_domain  FVCOM domain name, one of leofs, lmhofs, loofs, and lsofs
  schism_grid   schism horizontal grid .nc file containing schism coordinates
  start_time    Start time of the SCHISM simulation period in a string that is supported by the Python dateutil.parser package, for example, "2024-01-02 00:00"  
  end_time      End time of the SCHISM simulation period in a string that is supported by the Python dateutil.parser package, for example, "2024-01-02 23:00"
  output        Output schism tidal boundary condition .nc file

options:
  -h, --help    show this help message and exit

#### Examples ####

python makeOcean_fvcom.py ../FVCOM_Download/data/loofs loofs ../Lake_Ontario/SCHISM/hgrid.gr3  "2024-01-02 00:00"  "2024-01-02 18:00" ../LO_elev2Dth.nc
