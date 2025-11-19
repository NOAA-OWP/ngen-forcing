#!/usr/bin/env bash
#

#source ./run_sing_coastal_workflow_set_env.bash

if [[ ! -d $DATAexec ]]; then
   mkdir -p $DATAexec
fi
source ./pre_nwm_forcing_coastal.bash

pre_nwm_forcing_coastal ${STARTPDY}${STARTCYC} \
	$DATAexec/coastal_forcing_output \
	$FCST_LENGTH_HRS \
        $NWM_FORCING_DIR




