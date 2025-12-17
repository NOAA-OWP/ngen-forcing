# Overview
This directory contains Python scripts for downloading forcing data from archived NWM AnA data and retrospective data.
NWM retrospective: https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/index.html
NWM AnA: https://console.cloud.google.com/storage/browser/national-water-model

# Script description
main.py : the main function
data_downloader.py: the download functions

# Script Usage
The script has a help option (-h) for printing usage information.

usage: main.py [-h]
               output_dir domain start_time end_time meteo_source hydrology_source
               coastal_water_level_source

positional arguments:
  output_dir            output directory
  domain                domain name, one of hawaii, prvi, pacfic, or atlgulf
  start_time            Start time of the SCHISM simulation period in a string that is supported by
                        the Python dateutil.parser package, for example, "2024-01-02 00:00"
  end_time              End time of the SCHISM simulation period in a string that is supported by
                        the Python dateutil.parser package, for example, "2024-01-02 23:00"
  meteo_source          nwm_retro or nwm_ana
  hydrology_source      nwm or ngen
  coastal_water_level_source
                        stofs, tpxo, glofs (great lakes only)

options:
  -h, --help            show this help message and exit


#### Examples ####

python main.py "/efs/coastal_testdata/data_downloader_output_pr_retro" \
	        "prvi" \
                "2022-09-13T00-00-00Z" \
                "2022-09-13T12-00-00Z"  \
		"nwm_retro" \
		"nwm" \
	        "stofs"	
