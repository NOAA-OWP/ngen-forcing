#!/bin/bash

# Downloads the unmasked SNODAS data files from https://noaadata.apps.nsidc.org
# Usage is described in the usage function.
set -e

BASE_URL=https://noaadata.apps.nsidc.org/NOAA/G02158/unmasked/

usage()
{
    echo "Usage: $0 [YEAR as yyyy] [MONTH as mm] [DAY as dd]"
    echo "  Providing no options will download SNODAS data for all years."
    echo "  Providing YEAR, or YEAR and MONTH, or YEAR and MONTH and DAY, will download SNODAS data for the specified year or year/day or year/month/day."
    echo ""
    echo "Examples:"
    echo "  snodas_downloader.sh --help"
    echo "  snodas_downloader.sh" 
    echo "  snodas_downloader.sh 2024"
    echo "  snodas_downloader.sh 2024 01"
    echo "  snodas_downloader.sh 2024 01 20"
}

if [[ $1 == "--help" ]] || [[ $1 == "-h" ]]; then
    usage
    exit 0
fi

year=$1
month=$2
day=$3
url=${BASE_URL}

# Determine the download URL based on the command arguments
if [[ -n $year ]]; then
    url="${url}${year}/"

    if [[ -n $month ]]; then
        month=$(printf "%02d" $month)
        month_mmm=$(date -d ${year}${month}01 +%b)
        url="${url}${month}_${month_mmm}/"

        if [[ -n $day ]]; then
            day=$(printf "%02d" $day)
            url="${url}SNODAS_unmasked_${year}${month}${day}.tar"
        fi
    fi
fi

# Download the data to a new directory named "snodas". The subdirectories under this directory mirror the directory structure on the NSIDC server. 
echo "Downloading SNODAS files ..."
wget  --recursive --no-host-directories --cut-dirs=2 --no-check-certificate --reject "index.html*" --no-parent --execute robots=off --directory-prefix snodas --no-verbose --show-progress ${url}

# Untar the all the daily files
echo "Untaring SNODAS files ..."
find ./snodas -type f -name "*.tar" -execdir tar -xf {} \;

# Remove all files not related to the Snow Water Equivalent parameter (product code 1034)
echo "Removing all files except Snow Water Equivalent (SWE) ..."
find ./snodas -type f -not -name "*1034*" -execdir rm --force {} \;

# Unzip the compressed SWE files ...
echo "Uncompressing the SWE files ..."
find ./snodas -type f -name "*1034*.gz" -execdir gunzip --force {} \;

echo "Download complete"

