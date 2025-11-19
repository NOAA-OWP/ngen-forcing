#!/usr/bin/env bash 

#--------------------------------------------------------------
#This task creates the initial coastal discharge using CHRT files found in:
#
#  $COMOUT/${CASETYPE}
#
#Addtionally, for forecast cycles, CHRT files are used from:
#
#  $COMOUT/$NWM_RESTART_CYCLE
#
#Output is written to vsource.th, vsink.th, and source_sink.in files in the working directory:
# 
#  $DATAexec
#
nwm_coastal_initial_discharge() {

   PDYcyc=$1
   PDY=${PDYcyc:0:8}
   cyc=${PDYcyc:8}
   let cycle_length_hrs="$2 - 1"
   nwm_cycle=$3
   nwm_retro_dir=$4

   local _file;

   #
   #This is a forcast cycle
   #
   nwm_ana_dir=${WRF_HYDRO_ROOT}_ana
   mkdir -p ${nwm_ana_dir}
   mkdir -p ${WRF_HYDRO_ROOT}

   for ((i=0; i<$((${cycle_length_hrs} + 1)); i++))
   do
       #pdycyc=$($NDATE $i $PDY$cyc)
       pdycyc=$(${USHnwm}/utils/advance_time.sh $PDY$cyc $i)
       if [[ $i -eq 0 ]]; then
         if [[ ${nwm_cycle} =~ "hawaii" ]]; then
            #copy the first from AnA
            _file=${pdycyc}'00.CHRTOUT_DOMAIN1'
            ln -sf ${nwm_retro_dir}/${_file} ${nwm_ana_dir}/${pdycyc}'00.CHRTOUT_DOMAIN1'

            for j in 15 30 45; do
              _file=${pdycyc}${j}'.CHRTOUT_DOMAIN1'
              ln -sf ${nwm_retro_dir}/${_file} ${WRF_HYDRO_ROOT}/${pdycyc}${j}'.CHRTOUT_DOMAIN1'
	    done
         else
            _file=${pdycyc}'00.CHRTOUT_DOMAIN1'
            ln -sf ${nwm_retro_dir}/${_file} ${nwm_ana_dir}/${pdycyc}'00.CHRTOUT_DOMAIN1'

         fi
       else
         if [[ ${nwm_cycle} =~ "hawaii" ]]; then
           #
           #hawaii time step is 15 minutes
           #
            for j in 00 15 30 45; do
              _file=${pdycyc}${j}'.CHRTOUT_DOMAIN1'
  	      ln -sf ${nwm_retro_dir}/${_file} ${WRF_HYDRO_ROOT}/${pdycyc}${j}'.CHRTOUT_DOMAIN1'
	    done
         else
            _file=${pdycyc}'00.CHRTOUT_DOMAIN1'
  	    ln -sf ${nwm_retro_dir}/${_file} ${WRF_HYDRO_ROOT}/${pdycyc}'00.CHRTOUT_DOMAIN1'
         fi
       fi
   done
   #
   #the last one
   #
   let totalcount="${cycle_length_hrs} + 1"
   #pdycyc=$($NDATE ${totalcount} $PDY$cyc)
   pdycyc=$(${USHnwm}/utils/advance_time.sh $PDY$cyc ${totalcount})
   if [[ ${nwm_cycle} =~ "hawaii" ]]; then
         _file=${pdycyc}'00.CHRTOUT_DOMAIN1'
	 ln -sf ${nwm_retro_dir}/${_file} ${WRF_HYDRO_ROOT}/${pdycyc}'00.CHRTOUT_DOMAIN1'
   else
         _file=${pdycyc}'00.CHRTOUT_DOMAIN1'
	 ln -sf ${nwm_retro_dir}/${_file} ${WRF_HYDRO_ROOT}/${pdycyc}'00.CHRTOUT_DOMAIN1'
   fi

  export NWM_ANA_DIR=${nwm_ana_dir}
  cp ${PARMnwm}/coastal/$COASTAL_DOMAIN/nwmReaches.csv $DATAexec

  python -u $USHnwm/wrf_hydro_workflow_dev/coastal/makeDischarge.py >> $DATAlogs/${nwm_cycle}.${PDY}${cyc}.log 2>&1

}
