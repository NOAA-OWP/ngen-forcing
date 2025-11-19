#!/usr/bin/bash

set -x

#--------------------------------------------------------------
#This task regrids ESTOFS forecasts to the coastal domain, and fills in water level forecasts as needed for the
#cycle length, using pytides.
#
#If $REGRIDDED_ESTOFS_FILE is populated, the task just copies this file to the output file, which is:
#
#   $DATAexec/elev2D.th.nc
#
#If $COASTAL_ESTOFS_FILE is populated, the task regrids this file:
#
#Otherwise, it looks in $$COMINestofs/estofs.YYYYMMDD for the closest file within 6 hours.
#
#If no ESTOFS file can be found, the task fails.
#
pre_nwm_coastal_regrid_estofs() {
   #
   # regrid_estofs needs gcc not intel
   #
#   module switch intel/${intel_ver} gcc/${gcc_ver}

#   start_date=${FORCING_BEGIN_DATE:0:8}
#   start_hour=${FORCING_START_HOUR}
   
   PDY=${1:0:8}
   cyc=${1:8}
   export LENGTH_HRS=$2
   export COASTAL_ESTOFS_FILE=$3

   start_date=${PDY}
   start_hour=${cyc}
   #estofs_data=$DATAexec/estofs.t${start_hour}z.fields.cwl.nc
   estofs_data=$DATAexec/stofs_2d_glo.t${start_hour}z.fields.cwl.nc
   output_file=$DATAexec/elev2D.th.nc

   if [[ "$COASTAL_ESTOFS_FILE" != "" ]]; then
           ln -sf $COASTAL_ESTOFS_FILE $estofs_data
           diffhrs=0

   #	
   #no mapping for extend AnA,  Long AnA, or medium range
   #
   elif [[ "${LENGTH_HRS}" -lt -6 || "${CASETYPE}" =~ "medium_range_" ]]; then
          diffhrs=6
          estofs_file=$COMINestofs/stofs_2d_glo.${start_date:0:8}/stofs_2d_glo.t${start_hour}z.fields.cwl.nc
          ln -sf $estofs_file $estofs_data
   else
           estofs_file=""
           best=""

           diffhrs=0
           if [ ${cyc} -lt "06" ]; then
              estofs_file=$COMINestofs/stofs_2d_glo.${PDYm1}/stofs_2d_glo.t18z.fields.cwl.nc
              t1=`date -u -d "${PDY} ${cyc}" +%s`
              t2=`date -u -d "${PDYm1} 18" +%s`
              diff=$((t1-t2))
              diffhrs=$((diff/3600))
           elif [ ${cyc} -ge "06" ] && [ ${cyc} -lt "12" ]; then
              estofs_file=$COMINestofs/stofs_2d_glo.${PDY}/stofs_2d_glo.t00z.fields.cwl.nc
              t1=`date -u -d "${PDY} ${cyc}" +%s`
              t2=`date -u -d "${PDY} 00" +%s`
              diff=$((t1-t2))
              diffhrs=$((diff/3600))
           elif [ ${cyc} -ge "12" ] && [ ${cyc} -lt "18" ]; then
              estofs_file=$COMINestofs/stofs_2d_glo.${PDY}/stofs_2d_glo.t06z.fields.cwl.nc
              t1=`date -u -d "${PDY} ${cyc}" +%s`
              t2=`date -u -d "${PDY} 06" +%s`
              diff=$((t1-t2))
              diffhrs=$((diff/3600))
           else
              estofs_file=$COMINestofs/stofs_2d_glo.${PDY}/stofs_2d_glo.t12z.fields.cwl.nc
              t1=`date -u -d "${PDY} ${cyc}" +%s`
              t2=`date -u -d "${PDY} 12" +%s`
              diff=$((t1-t2))
              diffhrs=$((diff/3600))
           fi

           ln -sf $estofs_file $estofs_data
   fi

#       $USHnwm/utils/waitFile.sh ${estofs_file} $waitTime
#
#       if [[ ! -f ${estofs_file} ]]; then
#	    err_exit "ESTOFS ${estofs_file} file doesn't exist." 
#       fi
       #cpfs $estofs_file $estofs_data

       hgrid_file=$DATAexec/open_bnds_hgrid.nc
       ln -sf ${PARMnwm}/coastal/$COASTAL_DOMAIN/open_bnds_hgrid.nc $hgrid_file
       #cpfs ${PARMnwm}/coastal/$COASTAL_DOMAIN/open_bnds_hgrid.nc $hgrid_file

       export ESTOFS_INPUT_FILE=$estofs_data
       export OPEN_BNDS_HGRID_FILE=$hgrid_file
       export SCHISM_OUTPUT_FILE=$output_file

       export CYCLE_DATE=$start_date
       #export CYCLE_TIME=$start_time
       export CYCLE_TIME=$cyc'00'
       local _old_length_hrs=${LENGTH_HRS}
       export LENGTH_HRS=$(( ${LENGTH_HRS/#-} + 1 )) 
}

