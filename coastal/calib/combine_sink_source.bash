#!/usr/bin/env bash

#--------------------------------------------------------------
#This task combines sink and source data found in:
#
#   $DATAexec by calling combine_sink_source.F90
#
#and writes output to:
#
#   $DATAexec
#
nwm_coastal_combine_sink_source() {
    cd $DATAexec
    cp  ${EXECnwm}/combine_sink_source ./
    printf '1\n2\n' | ./combine_sink_source
}


