#!/usr/bin/bash

set -x

make_tpxo_ocean() {
   #
   # create water level boundary file from TPXO data 
   #

#   start_date=${FORCING_BEGIN_DATE:0:8}
#   start_hour=${FORCING_START_HOUR}
   
   PDY=${1:0:8}
   cyc=${1:8}
   export LENGTH_HRS=$2
   export OTPSnc_DIR=$3
   export NGEN_FORCING_DIR=$4
   export SCHISM_PARM_DIR=$5
   export COASTAL_DOMAIN=$6
   export TIME_STEP_IN_SECS=$7

#   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/contrib/software/gcc/8.5.0/lib64:/contrib/software/netcdf/4.7.4/lib

   export END_DATETIME=$(${USHnwm}/utils/advance_time.sh $PDY$cyc $LENGTH_HRS)
   python $NGEN_FORCING_DIR/coastal/tidal/tpxo_to_open_bnds_hgrid/make_otps_input.py \
	  $SCHISM_PARM_DIR/prvi/open_bnds_hgrid.nc \
	  $PDY$cyc $END_DATETIME $TIME_STEP_IN_SECS  \
	  $DATAexec/otps_lat_lon_time.txt

   cp ./setup_tpxo.txt $DATAexec/
   cp ./Model_tpxo10_atlas $DATAexec/

   cd $DATAexec

#   ln -sf $OTPSnc_DIR/DATA ./
   ln -sf $NGWPC_COASTAL_PARM_DIR/TPXO10_atlas_v2_nc .

   $OTPSnc_DIR/predict_tide < setup_tpxo.txt

   python $NGEN_FORCING_DIR/coastal/tidal/tpxo_to_open_bnds_hgrid/otps_to_open_bnds_hgrid.py  \
	  ./otps_out.txt                                                               \
          $SCHISM_PARM_DIR/$COASTAL_DOMAIN/open_bnds_hgrid.nc                          \
	  ./elev2D.th.nc

   local _correction_file=$DATAexec/elevation_correction.csv
   if [[ -f  ${_correction_file} ]]; then
       echo "Applying elevation datum correction to elev2D.th.nc file"
       ${MPICOMMAND3} python $USHnwm/wrf_hydro_workflow_dev/coastal/correct_elevation.py \
       ./elev2D.th.nc ${_correction_file} 
   fi
   cd -
}


