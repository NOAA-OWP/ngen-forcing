# Overview
This directory contains scripts for downloading and processing SNODAS data from the National Snow and Ice Data Center (NSIDC).

# Script Description
snodas_downloader.sh: This script downloads unmasked SNODAS data from the NSIDC server. Only the Snow Water Equivalent files are saved.

snodas_convert.py: A rough but portable utility script for converting binary SNODAS files to NetCDF. Assumes use with snodas_downloader.sh. 

# Script Usage
#### snodas_downloader.sh
snodas_downloader.sh [-h|--help] [YEAR as yyyy] [MONTH as mm] [DAY as dd]

Providing no options will download SNODAS data for all years.
Providing YEAR, or YEAR and MONTH, or YEAR and MONTH and DAY, will download SNODAS data for the specified year or year/day or year/month/day.
-h, --help: Prints usage 

#### snodas_convert.py

python snodas_convert.py

This script needs to be run from the unmasked/ directory (used by snodas_downloader.sh). Either move or copy this script to that directory. It also assumes the directory strcture matches that used by the snodas_downloader.sh script. 

Ensure that Docker is available/active, and that you have the correct permissions to use it. The script will use a GDAL image, pulling it if necessary. Using a container image is intended to help with portability, and isolate GDAL to limit potential conflicts.

This script is configured only for use with SNODAS data from 01OCT2013 or later. Older data has already been archived, but requires different GDAL settings.

# Examples
#### snodas_downloader.sh
snodas_downloader.sh --help
snodas_downloader.sh
snodas_downloader.sh 2024
snodas_downloader.sh 2024 01
snodas_downloader.sh 2024 01 20
#### snodas_convert.py
python snodas_convert.py


