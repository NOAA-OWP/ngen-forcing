#!/usr/bin/env bash

set -x

#--------------------------------------------------------------
#This task merges source and sink data found in:
#
#   $DATAexec
#
#Using source_sink, vsource, and vsink data found in:
#
#   $DATAexec
#
nwm_coastal_merge_source_sink() {
    nwm_ensemble_mem=$1
    nwm_cycle=$2
    nwm_base_cycle=$3
   
    cd $DATAexec

#    if [[ "${nwm_ensemble_mem}" == "" ]]; then
#        cycle=$nwm_cycle
#    else
#        cycle=${nwm_cycle}_mem${nwm_ensemble_mem}
#    fi
    cycle=$nwm_cycle

    if [[ $nwm_base_cycle != $cycle ]]; then
        #ln -fs ${PARMnwm}/coastal/$COASTAL_DOMAIN/source_sink* $DATAexec/.
        #ln -fs ${PARMnwm}/coastal/$COASTAL_DOMAIN/vsource* $DATAexec/.
        #ln -fs ${PARMnwm}/coastal/$COASTAL_DOMAIN/vsink* $DATAexec/.
        ln -fs ${PARMnwm}/coastal/$COASTAL_DOMAIN/source.nc $DATAexec/.
    fi

    local _coastal_root_dir=$COASTAL_ROOT_DIR
    export COASTAL_ROOT_DIR=$DATAexec
    export COASTAL_WORK_DIR=$DATAexec
    python $USHnwm/wrf_hydro_workflow_dev/coastal/merge_source_sink.py >> $DATAlogs/merge_source_sink.log 2>&1
    export COASTAL_ROOT_DIR=$_coastal_root_dir

}

