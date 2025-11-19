# Overview
This directory contains Python scripts for pre- and post-processing input and output files for OTPSnc. The OTPSnc program calculates the sea water levels at given locations and times using the OSU TPXO Tide Models (https://www.tpxo.net/home). 

The post-processisng script uses output of the OTPSnc program to create the SCHISM elev2D.th.nc file. SCHISM uses this file to set the boundary conditions. 


# OTPSnc
Information about OTPSnc can be found here, https://www.tpxo.net/otps.

# Prerequisite
Both the pre- and post-processing script need the SCHISM parameter file, open_bnds_hgrid.nc, as an input.   


# Script description
TPXOOut.py : The class definition for the OTPSnc tide prediction output file.
make_otps_input.py : create the OTPSnc/predict_tide input file for a given open_bnds_hgrid.nc file, time period and time step. 
otps_to_open_bnds_hgrid.py: the post-processing script that creates a SCHISM elev2D.th.nc file using the OTPSnc/predict_tide output file.
tpxo_output.py : This script makes an open_bnds_hgrid.nc file for a given list of node locations in lat/lon coordinates. It requires an existing open_bnds_hgrid.nc as a template. It is provided here as a convenient tool when an open_bnds_hgrid.nc file is not available.

# Script Usage
Each script has a help option (-h) for printing usage information.

make_otps_input.py:

   Syntax to run:
      make_otps_input.py schism_grid start_time end_time time_step output 

      positional arguments:
           schism_grid  .nc file containing schism coordinates
           start_time   start time of the timeseries, YYYYmmddHH
           end_time     end time of the timeseries, YYYYmmddHH
           time_step    time step in seconds
           output       the OTPSnc input filename

   Accepted formats for start_time and end_time include:
    YYYYmmddHH

otps_to_open_bnds_hgrid.py:

   Syntax to run:
      otps_to_open_bnds_hgrid.py otpsnc_output schism_grid output

      positional arguments:
         otpsnc_output  Input waterlevel timeseries file
         schism_grid    .nc file containing schism coordinates
         output         Output .nc file

tpxo_output.py: edit the source directly 

#### Examples ####

    ./make_otps_input.py ./nwm.v3.0.6_no_svn/parm/coastal/pacific/open_bnds_hgrid.nc  2023040100 2023040123 3600 pacific_20230401_lat_lon_time

   ./otps_to_open_bnds_hgrid.py  ./lower_co.out ./lco_open_bnds_bhgrid.nc elev2D.th.nc

