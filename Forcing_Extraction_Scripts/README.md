# Overview
This directory contains a series of scripts for each NWM domain subdirectory (CONUS, Alaska, Puerto Rico, Hawaii) that encompasses the required meteorological forcing data products needed for each regional NWMv3.0 operational configuration setup. Each script is a particular meteorlogical forcing data product that is available to download off the NOMADS server. Availability of each meteorlogical forcing data product varies, but a user can generally extract at least the last 24 hours of previous data products or forecast cycles available. 

# Setting Up Required Python Environment to Execute Forcing Extraction Scripts Using Anaconda
conda env create --name forcing_extraction --file=environment.yml

# Script Execution Example and Argument Descriptions

#### Example ####
python get_prod_GFS.py ./output --lookBackHours=24 --cleanBackHours=36 --lagBackHours=6

#### Arguments to the Forcing Extraction Scripts ####
* outDir (required) - Output directory pathway where the NOMADS data will be downloaded to
* lookBackHours (optional) - How many hours to look back for forecast data cycles
* cleanBackHours (optional) - Period between current time and the beginning of the lookback period to cleanout old data
* lagBackHours (optional) - Wait at least this long back before searching for files


#### Things to Keep in Mind ####
* MRMS precipitation products only store the last 24 hours of satellite swath data.
* At times, the server lags and scripts will take between 2-5 minutes to finally recieve the url request and download a given file.
* National Blended Model (NWM) products do not have a presistent forecast cycle output file interval. Future changes will be coming however for NBM forecast products to always have hourly data.
* HRRR data product output varies from region to region (Hourly - CONUS, 3-hourly Alaska)
* AORC data downloading is not available in this repository currently for CONUS and Alaska domains. The Office of Water Prediction (OWP) is working on making a public data respository for these data products. Once they go online, we will update this repsoitory accordingly. 
