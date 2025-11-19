#/usr/bin/evn bash

set -x

#--------------------------------------------------------------
#This task runs the coastal forcing engine, which creates coastal forcing data using LDASIN files found in:
#
#   ${forcing_atm} e.g. $GESIN/analysis_assim
#
#using the geogrid file at:
#
#  $PARMnwm/domain_$NWM_DOMAIN/$GEOGRID_FILE
#
#for $LENGTH_HRS hours beginning at $CYCLE_DATE $CYCLE_TIME (or ending at this time, if using a negative lookback).
#
#Coastal orcing output (vsource.th.2, source_sink.in.2, msource.th.2) is written to:
#
#   $COASTAL_FORCING_OUTPUT_DIR, e.g.  ${GESOUT}/${CASETYPE}
#
#and symlinked to:
#
#   $DATAexec
#
pre_nwm_forcing_coastal() {

  PDY=${1:0:8}
  cyc=${1:8}
  export COASTAL_FORCING_OUTPUT_DIR=$2
  export LENGTH_HRS=$3
  nwm_forcing_retro_dir=$4

  local _file;

  export FORCING_BEGIN_DATE=${PDY}${cyc}00
  export NWM_FORCING_OUTPUT_DIR=$DATAexec/forcing_input
  export FORCING_END_DATE=$(${USHnwm}/utils/advance_time.sh $PDY$cyc $LENGTH_HRS)'00'
  export COASTAL_FORCING_INPUT_DIR=$NWM_FORCING_OUTPUT_DIR/${FORCING_BEGIN_DATE:0:10}
  export COASTAL_WORK_DIR=$DATAexec
  export FORCING_START_YEAR=${PDY:0:4}
  export FORCING_START_MONTH=${PDY:4:2}
  export FORCING_START_DAY=${PDY:6:2}
  export FORCING_START_HOUR=${cyc}

  #
  #Get t=0 data from AnA cycle in $COM to avoid conflicts with later AnA forcing jobs  
  #
  #Forcing AnA jobs will keep updating the forcing files and causes concurrent problems
  #

  #$USHnwm/utils/waitFile.sh ${forcing_atm_ana}/${FORCING_BEGIN_DATE}.LDASIN_DOMAIN1 $waitTime
  #
  mkdir -p $NWM_FORCING_OUTPUT_DIR/${FORCING_BEGIN_DATE:0:10}

  pdycyc=${PDY}${cyc}
  for ((i=0; i<=${LENGTH_HRS}; i++))
  do
        f=${pdycyc}.LDASIN_DOMAIN1
        ln -sf ${nwm_forcing_retro_dir}/${f} $NWM_FORCING_OUTPUT_DIR/${FORCING_BEGIN_DATE:0:10}
	let hr="$i + 1"
	pdycyc=$(${USHnwm}/utils/advance_time.sh $PDY$cyc $hr)
  done

  mkdir -p $COASTAL_FORCING_OUTPUT_DIR

  export FECPP_JOB_INDEX=0
  export FECPP_JOB_COUNT=1
}
