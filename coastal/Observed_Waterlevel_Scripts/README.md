# Overview
This directory contains Python scripts for downloading observed water level data from the NOAA CO-OPS server.


# NOAA CO-OPS APIs
The documentation about NOAA CO-OPS APIs can be found here,

NOAA CO-OPS for data retrival:
https://api.tidesandcurrents.noaa.gov/api/prod/

CO-OPS Metadata API:
https://api.tidesandcurrents.noaa.gov/mdapi/prod/#intro

# Script description
find_waterlevel_stations.py: get a list of the water level stations. Executed by download_noaa_obv_wl.py
download_noaa_obv_wl.py: First call find_waterlevel_stations.py and then iterate through the station list and download water level data using the begin and end time and other options given by the user

# Script Usage
Each script has a help option (-h) for printing usage information.

   Syntax to run:
      download_noaa_obv_wl.py -o <outputdir> -b <begin_date> -e <end_date> -m 
   where -o defines the ouput directory
         -b defines the begin time for the timeseries
         -e defines the end time for the timeseries
         -l is optional. Defines a list of station ids seperated by ','. If not given, 
            data for all avaialble stations will be downloaded. 
         -m is optional. If not given, hourly height will be downloaded, 
            otherwise, 6 minute inteval water leel data will be downloaded.

   Accepted formats for begin and end time include:
    yyyyMMdd
    yyyyMMdd HH:mm
    MM/dd/yyyy
    MM/dd/yyyy HH:mm

#### Examples ####

   python download_noaa_obv_wl.py -o "./testdir" -b 20250401 -l 1611400,1612480,1617760,1612340,1615680,1612401,1617433  -e 20250402 -m
   python download_noaa_obv_wl.py -o "./testdir" -b 20250401 -l 1611400,1612480,1617760,1612340,1615680,1612401,1617433  -e 20250402
   python download_noaa_obv_wl.py -o "./testdir" -b 20241101 -e 20241112
