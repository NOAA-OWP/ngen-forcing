#/usr/bin/env bash

pdy=20250511
mkdir -p /efs/$LOGNAME/schism_forecast_test/hrrr/hrrr.${pdy}
cd /efs/$LOGNAME/schism_forecast_test/hrrr/hrrr.${pdy}
for j in {00..23}; do
  for i in {00..48}; do
    wget -nc --no-check-certificate https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.${pdy}/conus/hrrr.t${j}z.wrfsfcf${i}.grib2
  done
done
