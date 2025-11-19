# Overview
This directory contains the bash script for downloading the Great Lakes' FVCOM model real-time and archived forecast and nowcast water level data from the NODD AWS cloud. This product is operated by NOAA's Center for Operational Oceanographic Products and Services.


# Great Lakes' FVCOM
The Great Lakes' FVCOM models are a part of the Operational Forecast System (OFS). There are four models to cover the entire Great Lakes domain, namely, LEOFS, LMHOFS, LOOFS, and LSOFS. The web-page https://tidesandcurrents.noaa.gov/models.html lists all OFS domains. Detailed information about Great Lakes' domains can be found by following the links in this domain.


# Product dissemination
Currently the products from National Ocean Service's operational forecast systems (OFS) are disseminated by both the NODD AWS cloud (https://noaa-nos-ofs-pds.s3.amazonaws.com/index.html) and  CO-OPS THREDDS server (https://opendap.co-ops.nos.noaa.gov/thredds/catalog.html). The NODD AWS cloud has both the archived and real-time output, while the CO-OPS THREDDS server has only data from the last two months including the real-time outputs. For simplicity, this download script uses the NODD AWS cloud to access the archived and real-time files.

# Script description
The script downloads the nowcast and forecast products for all of the four Great Lakes models for a given date. 

For each of the nowcast or forecast product, there are three type of output files - station, grid, and field. The total number of file types is 6 as listed below.
- station nowcast
- station forecast
- grid nowcast
- grid forecast
- field nowcast
- field forecast

# Script Usage
the script has a help option (-h) for printing usage information.

Usage: ./download_fvcom.bash [-s <start_utcdate>(yyyymmdd)]  [-e <end_utcdate>(yyyymmdd)] [-n <domain> (one of leofs, lmhofs, loofs, or lsofs)] [-o <output path>]
        defaults: 
                <start_utcdate>: current utc day
                <end_utcdate>: current utc day + 1 day
                <domain>: all 4 domains
                <output path>: $ROOT_SHARE/data)

   where <start_utcdate>  is the UTC date such as 20241218, default is the current date
         <domain> is one of the Great Lakes' domains, one of leofs, lmhofs, loofs, or lsofs, default is to download all domains
         <output path> the output directory where the downloaded files will be saved. The default value is $ROOT_SHARE/data. ROOT_SHARE is an environmental variable. If the <output path> is not given and the ROOT_SHARE environmental variable is not defined, the script will exit with an error message.

#### Examples ####

   ./download_fvcom.bash -s 20241218 -e 20241220 -o ./fvcom_data
   ROOT_SHARE=./fvcom_data ./download_fvcom.bash
   ./download_fvcom.bash -s 20241218 -e 20241220 -n loofs -o ./fvcom_data
