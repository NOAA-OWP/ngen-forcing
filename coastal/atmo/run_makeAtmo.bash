#!/usr/bin/env bash

export COASTAL_FORCING_INPUT_DIR=/lustre/Zhengtao.Cui/short_range_coastal_pacific_2024022013/forcing_input/2024022013/
export LENGTH_HRS=18
export FORCING_BEGIN_DATE=202402201300
export FORCING_END_DATE=202402210700
export COASTAL_FORCING_OUTPUT_DIR=/lustre/Zhengtao.Cui/test/tmp/nwm_analysis_assim_coastal_hawaii_13_v3.0/coastal_forcing_output
export COASTAL_FORCING_OUTPUT_DIR=/lustre/Zhengtao.Cui/coastal_forcing_output
export GEOGRID_FILE=/lustre/Zhengtao.Cui/test/packages/nwm.v3.0.6/parm/domain/geo_em_CONUS.nc

export COASTAL_WORK_DIR=/lustre/Zhengtao.Cui/sfincs_atmo
export FORCING_START_YEAR=2024
export FORCING_START_MONTH=02
export FORCING_START_DAY=20
export FORCING_START_HOUR=13

mkdir -p $COASTAL_WORK_DIR/sflux/

python -u .//makeAtmo.py 

