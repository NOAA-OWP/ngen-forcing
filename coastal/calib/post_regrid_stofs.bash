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
post_nwm_coastal_regrid_estofs() {
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


   diffhrs=0
   # if we are Medium-Range, use PyTides to fill in the water levels for hours 181-241
   if [ ${LENGTH_HRS} -gt $((180-$diffhrs)) ]; then
           old_length_hrs=$LENGTH_HRS
           export LENGTH_HRS=$(($LENGTH_HRS+$diffhrs))
           export TIDAL_CONSTANTS_DIR=$COASTAL_ROOT_DIR/Tides/TidalConst
           export COASTAL_DOMAIN_GR3=$PARMnwm/coastal/$COASTAL_DOMAIN/hgrid.gr3

           python $USHnwm/wrf_hydro_workflow_dev/coastal/Tides/makeOceanTide.py >> $DATAlogs/regrid_stofs.{PDY}${cyc}.log 2>&1
           export LENGTH_HRS=${old_length_hrs}

   fi

   local _correction_file=$DATAexec/elevation_correction.csv
   if [[ -f  ${_correction_file} ]]; then
           echo "Applying elevation datum correction to elev2D.th.nc file"
           python $USHnwm/wrf_hydro_workflow_dev/coastal/correct_elevation.py \
		   $SCHISM_OUTPUT_FILE \
		    ${_correction_file} >> $DATAlogs/regrid_stofs.${PDY}${cyc}.log 2>&1
   fi

   #
   #switch back to intel
#   module switch gcc/${gcc_ver} intel/${intel_ver} 
}

