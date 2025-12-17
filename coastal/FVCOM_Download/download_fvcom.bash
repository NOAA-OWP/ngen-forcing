#!/usr/bin/env bash

if [[ -z $ROOT_SHARE ]]; then
    ROOT_SHARE_TXT="ERROR: ROOT_SHARE variable not set!"
else
    ROOT_SHARE_TXT="ROOT_SHARE set to: $ROOT_SHARE"
fi

usage() { echo -e "Usage: $0 [-s <start_utcdate>(yyyymmdd)]  [-e <end_utcdate>(yyyymmdd)] [-n <domain> (one of leofs, lmhofs, loofs, or lsofs)] [-o <output path>]\n\tdefaults: \n\t\t<start_utcdate>: current utc day\n\t\t<end_utcdate>: current utc day + 1 day\n\t\t<domain>: all 4 domains\n\t\t<output path>: \$ROOT_SHARE/data)" 1>&2; exit 1; }

options=$(getopt -o s:e:n:o:h --long start_utcdate:,end_utcdate:,domain:,output:,help -- "$@")
eval set -- "$options"

while :; do
    case "$1" in
        -s|--start_utcdate)
            shift
            INPUT_DATE_START=$1
            echo "got option: $INPUT_DATE_START"
            ;;
        -e|--end_utcdate)
            shift
            INPUT_DATE_END=$1
            echo "got option: $INPUT_DATE_END"
            ;;
        -n|--domain)
            shift
            OPTION_DOMAIN=$1
            echo "got option: $OPTION_DOMAIN"
            ;;
        -o|--output)
            shift
            OPTION_DIR=$1
            echo "got option: $OPTION_DIR"
            ;;
        -h|--help)
            usage
            exit
            ;;
        --)
            shift
            break
            ;;
    esac

    shift
done

if [[ -z $INPUT_DATE_START ]]; then
    UTC_DATE_START=`date "+%Y%m%d"`
else
    echo "Date specified on cmd line... `date -d $INPUT_DATE_START "+%Y%m%d"`"
    UTC_DATE_START=`date -d $INPUT_DATE_START "+%Y%m%d"`
fi

if [[ -z $INPUT_DATE_END ]]; then
    UTC_DATE_END=`date '+%C%y%m%d' -d "+1 days""`
else
    echo "Date specified on cmd line... `date -d $INPUT_DATE_END "+%Y%m%d"`"
    UTC_DATE_END=`date -d $INPUT_DATE_END "+%Y%m%d"`
fi

if [[ -z $OPTION_DOMAIN ]]; then
    declare -a DOMAINS=(leofs lmhofs loofs lsofs)
    echo "Download all domains: leofs lmhofs loofs lsofs"
else
    echo "Domain specified on cmd line... $OPTION_DOMAIN"
    declare -a DOMAINS=($OPTION_DOMAIN)
fi
if [[ -z $OPTION_DIR ]]; then
    echo -e $ROOT_SHARE_TXT
    if [[ -z $ROOT_SHARE ]]; then
        exit 1
    fi
    OUTPUT_DIR="$ROOT_SHARE/data"
else
    OUTPUT_DIR="$OPTION_DIR"
fi

pdy=$UTC_DATE_START
recent_two_month=$(date -d "1 month ago" "+%Y%m")01

#=========================================================
#  https://noaa-nos-ofs-pds.s3.amazonaws.com/${domain}/netcdf/[YYYYMM]/
#  nowcast grid
#     ${domain}.t[00-18.6]z.[date].regulargrid.n[000..006].nc
#  nowcast field
#     ${domain}.t[00-18.6]z.[date].fields.n[000..006].nc
#  nowcast station
#     ${domain}.t[00-18.6]z.[date].stations.nowcast.nc
#  forecast grid
#     ${domain}.t[00-18.6]z.[date].regulargrid.f[000..006].nc
#  forecast field
#     ${domain}.t[00-18.6]z.[date].fields.f[000..006].nc
#  forecast station
#     ${domain}.t[00-18.6]z.[date].stations.forecast.nc
#=========================================================
# pre 202403 files
# nocast field
#   nos.{domain}.fields.n[000..006].[date].t[00-18.6]z.nc
# nocast stations
#   nos.{domain}.stations.nowcast.[date].t[00-18.6]z.nc
# nocast grid
#   N/A
# forecast field
#   N/A
# forecast stations
#   N/A
# forecast grid
#   N/A
#=========================================================
# pre 202301 files
# nocast field
#   nos.{domain}.fields.n[000..006].[date].t[00-18.6]z.nc
# nocast stations
#   N/A
# nocast grid
#   N/A
# forecast field
#   N/A
# forecast stations
#   N/A
# forecast grid
#   N/A
#=========================================================

declare -A domain_to_lake=( \
	   [leofs]="erie" \
	   [lmhofs]="michigan-and-huron" \
	   [loofs]="ontario" \
	   [lsofs]="superior" )

for domain in "${DOMAINS[@]}"; do
     
   OFSDIR=$OUTPUT_DIR/${domain}
   if [ ! -d "${OFSDIR}" ]; then 
        mkdir -p $OFSDIR
   fi
         
   cd $OFSDIR

   #URLBASE=https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/lake-${domain_to_lake[$domain]}-operational-forecast-system-${domain}/
   URLBASE=https://noaa-nos-ofs-pds.s3.amazonaws.com/${domain}/netcdf
  while [ $pdy -lt $UTC_DATE_END ]; do
   for cyc in 00 06 12 18; do
     #pre 202301
     if [ ${pdy} -lt 20230101 ]; then
        for i in {000..006}; do 
            wget -nc --no-check-certificate \
	        ${URLBASE}/${pdy:0:-2}/nos.${domain}.fields.n${i}.${pdy}.t${cyc}z.nc
	        #${URLBASE}/${pdy:0:4}/${pdy:4:2}/nos.${domain}.fields.n${i}.${pdy}.t${cyc}z.nc
        done
     
     #pre 202403
     elif [ ${pdy} -lt 20240301 ]; then
        for i in {000..006}; do 
            wget -nc --no-check-certificate \
	        ${URLBASE}/${pdy:0:-2}/nos.${domain}.fields.n${i}.${pdy}.t${cyc}z.nc
	        #${URLBASE}/${pdy:0:4}/${pdy:4:2}/nos.${domain}.fields.n${i}.${pdy}.t${cyc}z.nc
        done
        wget -nc --no-check-certificate \
	        ${URLBASE}/${pdy:0:-2}/nos.${domain}.stations.nowcast.${pdy}.t${cyc}z.nc
	        #${URLBASE}/${pdy:0:4}/${pdy:4:2}/nos.${domain}.stations.nowcast.${pdy}.t${cyc}z.nc
     else
#        if [ ${pdy} -lt $recent_two_month ]; then
#           #URL=https://noaa-nos-ofs-pds.s3.amazonaws.com/${domain}/netcdf/${pdy:0:-2}
#	   URL=https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/lake-${domain_to_lake[$domain]}-operational-forecast-system-${domain}/${pdy:0:4}/${pdy:4:2}
#        else
#           URL=https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/lake-${domain_to_lake[$domain]}-operational-forecast-system-${domain}/${pdy:0:4}/${pdy:4:2}
#        fi

        #URL=https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/lake-${domain_to_lake[$domain]}-operational-forecast-system-${domain}/${pdy:0:4}/${pdy:4:2}
        URL=https://noaa-nos-ofs-pds.s3.amazonaws.com/${domain}/netcdf/${pdy:0:-2}
	#nowcast station
        wget -nc --no-check-certificate \
	        ${URL}/${domain}.t${cyc}z.${pdy}.stations.nowcast.nc
	#nowcast regulargrid and grid
#        for i in {000..006}; do 
#            wget -nc --no-check-certificate \
#	        ${URL}/${domain}.t${cyc}z.${pdy}.regulargrid.n${i}.nc
#            wget -nc --no-check-certificate \
#	        ${URL}/${domain}.t${cyc}z.${pdy}.fields.n${i}.nc
#        done

	#forecast station
        wget -nc --no-check-certificate \
		${URL}/${domain}.t${cyc}z.${pdy}.stations.forecast.nc
	#forecast regulargrid and grid
#        for i in {000..120}; do 
#            wget -nc --no-check-certificate \
#	        ${URL}/${domain}.t${cyc}z.${pdy}.regulargrid.f${i}.nc
#            wget -nc --no-check-certificate \
#	        ${URL}/${domain}.t${cyc}z.${pdy}.fields.f${i}.nc
#        done
     fi
   done
   pdy=$(date '+%C%y%m%d' -d "$pdy+1 days") 
  done
  cd -

done

