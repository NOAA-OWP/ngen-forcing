#!/usr/bin/bash

set -x

source ./initial_discharge.bash
source ./combine_sink_source.bash
source ./merge_source_sink.bash

#--------------------------------------------------------------
#
#This task runs SCHISM on the prepared coastal inputs, combines output and restarts, and writes output to:
#
#   $DATAexec/outputs
#
pre_nwm_coastal() {

   if [[ ! -d $DATAexec/outputs ]]; then
       mkdir -p $DATAexec/outputs
   fi
   cd $DATAexec

   export NWM_CYCLE=forecast
   export WRF_HYDRO_ROOT=$DATAexec/nwm_output

   nwm_coastal_initial_discharge ${STARTPDY}${STARTCYC} $FCST_LENGTH_HRS $COASTAL_DOMAIN $NWM_CHROUT_DIR

   nwm_coastal_combine_sink_source

   export COASTAL_ROOT_DIR=$DATAexec

   nwm_coastal_merge_source_sink "" "forecast" "forecast"

   export NSCRIBES=2

   #create offline partition
   create_offline_partition $NPROCS "${NSCRIBES}"
   #cpfs ${EXECnwm}/pschism_wcoss2_NO_PARMETIS_TVD-VL .
   #cpfs ${EXECnwm}/pschism_mistral_NOPM_VL .
   cp ${EXECnwm}/pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi .

}

#--------------------------------------------------------------
#
# Create offline partition file for a given number of processors
#  and domain
function create_offline_partition() {
  local num_procs=$1
  local scribes=$2

  cp ${EXECnwm}/metis_prep ./
  cp ${EXECnwm}/gpmetis ./
  ./metis_prep ./hgrid.gr3 ./vgrid.in
  ./gpmetis ./graphinfo $((${num_procs} - ${scribes})) -ufactor=1.01 -seed=15
  awk '{print NR,$0}' graphinfo.part.$((${num_procs} - ${scribes})) > partition.prop
}

