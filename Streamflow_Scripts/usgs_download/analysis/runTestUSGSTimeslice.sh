#!/usr/bin/env bash
##########################################################
# Test usgs timeslise using xml or json format data file.#
# ./runTestUSGSTimeslice.sh <XML or JSON>                #
#                                                        #
# 06/12/2024 CPham                                       #
##########################################################
cd $(pwd)

if [[ ! -d test_data/usgs_timeslices$1 ]]; then
  mkdir -p test_data/usgs_timeslices$1
else
  rm -f test_data/usgs_timeslices$1/*
fi

python make_time_slice_from_usgs_waterml.py -i test_data/$1 -o test_data/usgs_timeslices$1
