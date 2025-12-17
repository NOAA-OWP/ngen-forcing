#!/usr/bin/bash

set -x

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

   #create offline partition
   create_offline_partition $NPROCS "${NSCRIBES}"
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

