#!/usr/bin/env bash

set -x 

#--------------------------------------------------------------
#This task updates coastal parameter files found in the 
# working directory:
#
#   $PARMnwm/coastal/param.nml
#
#and copies these static files:
#
#   $PARMnwm/coastal/$COASTAL_DOMAIN/hgrid.gr3
#   $PARMnwm/coastal/$COASTAL_DOMAIN/hgrid.ll
#   $PARMnwm/coastal/$COASTAL_DOMAIN/manning.gr3
#   $PARMnwm/coastal/$COASTAL_DOMAIN/vgrid.in
#   $PARMnwm/coastal/$COASTAL_DOMAIN/bctides.in
#   $PARMnwm/coastal/$COASTAL_DOMAIN/windrot_geo2proj.gr3
#   $PARMnwm/coastal/$COASTAL_DOMAIN/station.in
#   $PARMnwm/coastal/$COASTAL_DOMAIN/hgrid.utm
#   $PARMnwm/coastal/$COASTAL_DOMAIN/hgrid.cpp
#   $PARMnwm/coastal/$COASTAL_DOMAIN/hotstart.nc
#
#into the working directory:
#
#   $DATAexec
#
#to cold restart comment out the line with ihot and nhot
#
#
#Update delivered parameters
# python /lfs/h1/owp/nwm/noscrub/Zhengtao.Cui/test/packages/nwm.v3.0.0/ush/wrf_hydro_workflow_dev/coastal/gr3_2_esmf.py hgrid.gr3 hgrid.nc
# python /lfs/h1/owp/nwm/noscrub/Zhengtao.Cui/test/packages/nwm.v3.0.0/ush/wrf_hydro_workflow_dev/coastal/gr3_2_esmf.py --filter_open_bnds hgrid.gr3 open_bnds_hgrid.nc
# python /lfs/h1/owp/nwm/noscrub/Zhengtao.Cui/test/packages/nwm.v3.0.0/ush/wrf_hydro_workflow_dev/coastal/make_element_areas.py  hgrid.nc element_areas.txt
#
#
nwm_coastal_update_params() {

  PDY=${1:0:8}
  cyc=${1:8}
  coastal_domain=$2
  export LENGTH_HRS=$3
  local _hotstartfile=$4

  local _cold_restart=0
  coastal_param=$DATAexec/param.nml
  cp ${PARMnwm}/coastal/$coastal_domain/param.nml $coastal_param

  if [[ "$_hotstartfile" == "" ]]; then
	  _cold_restart=1
  fi

  if [[ $LENGTH_HRS -le 0 ]]; then
      let cycle_length="$LENGTH_HRS + 1"
      #export SCHISM_BEGIN_DATE=$($NDATE $cycle_length $PDY$cyc)'00'
      #export SCHISM_BEGIN_DATE=$($NDATE $LENGTH_HRS $PDY$cyc)'00'
      export SCHISM_BEGIN_DATE=$(${USHnwm}/utils/advance_time.sh $PDY$cyc $LENGTH_HRS)'00'
      export SCHISM_END_DATE=$PDY$cyc'00'
      #let rnhours=$LENGTH_HRS*-1-1
      let rnhours=$LENGTH_HRS*-1
      #let ihfskip=$rnhours*18
      let ihfskip=18
      let nspool=18
  else
      export SCHISM_BEGIN_DATE=$PDY$cyc'00'
      #export SCHISM_END_DATE=$($NDATE $LENGTH_HRS $PDY$cyc)'00'
      export SCHISM_END_DATE=$(${USHnwm}/utils/advance_time.sh $PDY$cyc $LENGTH_HRS)'00'
      #let rnhours="$LENGTH_HRS-1"
      let rnhours="$LENGTH_HRS-0"
      #
      #Medium Range outputs every 1 hours
      #
      #if [[ ${CASETYPE} =~ "medium_range" ]]; then
      #  let ihfskip=54
      #  let nspool=54
      #else
      let ihfskip=18
      let nspool=18
      #fi
  fi

  start_date=$SCHISM_BEGIN_DATE
  start_year=${start_date:0:4}
  start_month=${start_date:4:2}
  start_day=${start_date:6:2}
  start_hour=${start_date:8:2}
  start_minute=${start_date:10:2}

  # SCHISM uses a fractional hour vs a separate minute
  start_hour=$(echo "scale=2;$start_hour + ($start_minute/60.0)" | bc -l)

  sed -i "s|^  start_year .*|  start_year = ${start_year}|;" ${coastal_param}
  sed -i "s|^  start_month .*|  start_month = ${start_month}|;" ${coastal_param}
  sed -i "s|^  start_day .*|  start_day = ${start_day}|;" ${coastal_param}
  sed -i "s|^  start_hour .*|  start_hour = ${start_hour}|;" ${coastal_param}
 
  sed -i "s|^  nspool .*|  nspool = ${nspool}|;" ${coastal_param}
  sed -i "s|^  ihfskip .*|  ihfskip = ${ihfskip}|;" ${coastal_param}

  # use netCDF forcings
  sed -i "s|^  if_source .*|  if_source = -1|;" ${coastal_param}
  #sed -i "s|^  if_source .*|  if_source = 0|;" ${coastal_param}

  # rnday is fractional day, convert from hours
  sed -i "s|^  rnday .*|  rnday = $(echo "scale=8;$rnhours/24.0" | bc -l)|;" ${coastal_param}

  # use 200 second model timestep and 10 minute atmospheric timestep
  sed -i "s|^  dt .*|  dt = 200 |;" ${coastal_param}
  sed -i "s|^  wtiminc .*|  wtiminc = 600 |;" ${coastal_param}
#  sed -i "s|^  nspool .*|  nspool = 18 |;" ${coastal_param}
#  sed -i "s|^  ihfskip .*|  ihfskip = 18 |;" ${coastal_param}

  # hotstart output
  # use the hotstart from the previous run
  if [[ "$CHAINED_REANALYSIS" != "" ]]; then
     sed -i "s|^  nhot_write .*|  nhot_write = 18|;" ${coastal_param}
     sed -i "s|^  ihfskip .*|  ihfskip = 18|;" ${coastal_param}
     sed -i "s|^  nhot .*|  nhot = 1|;" ${coastal_param}
     restart_date=$CHAINED_REANALYSIS
     restart_hr=${restart_date:8:2}
     restart_dt=${restart_date:0:8}
     out=${COMOUT}/$NWM_RESTART_CYCLE/coastal/$coastal_domain/$CHAINED_REANALYSIS
     fname=$out/hotstart_${coastal_domain}_${restart_dt}_${restart_hr}00.nc
     if [[ -e $fname ]]; then
        sed -i "s|^  ihot .*|  ihot = 1|;" ${coastal_param}
        ln -sf $fname $DATAexec/hotstart.nc
     else
        sed -i "s|^  ihot .*|  ihot = 0|;" ${coastal_param}
     fi
  # this is a cold-start reanalysis cycle
  elif [[ "$NWM_RESTART_CYCLE" == "reanalysis" && $LENGTH_HRS > 0 ]]; then
     sed -i "s|^  ihot .*|  ihot = 0|;" ${coastal_param}
     sed -i "s|^  nhot .*|  nhot = 0|;" ${coastal_param}
     sed -i "s|^  nhot_write .*|  nhot_write = 72|;" ${coastal_param}
  # this is an analysis cycle
  elif [[ "$NWM_CYCLE" == "$NWM_RESTART_CYCLE" ]]; then
     if [[ "$_cold_restart" == "1" ]]; then
       sed -i "s|^  ihot .*|  ihot = 0|;" ${coastal_param}
     else
       sed -i "s|^  ihot .*|  ihot = 1|;" ${coastal_param}
     fi
     sed -i "s|^  nhot .*|  nhot = 1|;" ${coastal_param}
     sed -i "s|^  nhot_write .*|  nhot_write = 18|;" ${coastal_param}
     sed -i "s|^  nspool .*|  nspool = 18 |;" ${coastal_param}
     sed -i "s|^  ihfskip .*|  ihfskip = 18|;" ${coastal_param}
     #
     # find The hotstart in the previous day for cycles 00z and 01z
     #
     # removing leading 0s with ${cyc#0} otherwise it is interprated as an octal number
     #if [[ ${cyc#0} -lt 2 ]]; then
     #  out=${COMOUTm1}/restart_coastal
     #else
     #  out=${COMOUT}/restart_coastal
     #fi

     if [[ "$_cold_restart" == "0" ]]; then
       if [[ -f $_hotstartfile ]]; then
          echo "Found hotstart file $_hotstartfile for cycle $cyc."
          cp $_hotstartfile $DATAexec/hotstart.nc
       fi
     fi
  else
     # symlink in appropriate hotstart
     if [[ "$_cold_restart" == "1" ]]; then
       sed -i "s|^  ihot .*|  ihot = 0|;" ${coastal_param}
     else
       sed -i "s|^  ihot .*|  ihot = 1|;" ${coastal_param}
     fi
     #sed -i "s|^  nhot .*|  nhot = 1|;" ${coastal_param}

     #don't write hotstart
     sed -i "s|^  nhot .*|  nhot = 0|;" ${coastal_param}
    
     if [[ ${CASETYPE} =~ "medium_range" ]]; then
	 if [[ "${RESTART_WRITE_HR}" != "" ]]; then 
	    sed -i "s|^  nhot_write .*|  nhot_write = $((18*${RESTART_WRITE_HR}))|;" ${coastal_param}
         else
            sed -i "s|^  nhot_write .*|  nhot_write = 2160|;" ${coastal_param}
         fi
     elif [[ ${CASETYPE} =~ "short_range" ]]; then
         sed -i "s|^  nhot_write .*|  nhot_write = 162|;" ${coastal_param}
     else
	 if [[ "${RESTART_WRITE_HR}" != "" ]]; then 
	    sed -i "s|^  nhot_write .*|  nhot_write = $((18*${RESTART_WRITE_HR}))|;" ${coastal_param}
         else
           echo "Unknown CASETYPE: ${CASETYPE}!"
         fi
     fi

     if [[ "$_cold_restart" == "0" ]]; then
       if [[ ! -f ${_hotstartfile} ]]; then
          msg="ERROR: ${CASETYPE} ${PDY} ${cyc}z is missing hotstart file, ${_hotstartfile}. Exiting ..."
          echo -e "$msg\n"
       #   echo -e "$msg\n" | mail.py -v -s "ERROR: NWM ${CASETYPE} ${PDY} ${cyc}z is missing hotstart file." $maillist3
          err_exit "ERROR: cannot find hotstart file, ${_hotstartfile}, for $PDY$cyc. Exiting ..."

       else
         echo "Found preferred hotstart file $_hotstartfile for cycle $cyc."
         cp ${_hotstartfile} $DATAexec/hotstart.nc
       fi
       export err=$?
       err_chk
     fi
  fi 


# COPY STATIC FILES
ln -sf ${PARMnwm}/coastal/${coastal_domain}/hgrid.gr3 $DATAexec
ln -sf ${PARMnwm}/coastal/${coastal_domain}/hgrid.ll $DATAexec
ln -sf ${PARMnwm}/coastal/${coastal_domain}/manning.gr3 $DATAexec
ln -sf ${PARMnwm}/coastal/${coastal_domain}/vgrid.in $DATAexec
ln -sf ${PARMnwm}/coastal/${coastal_domain}/bctides.in $DATAexec
ln -sf ${PARMnwm}/coastal/${coastal_domain}/windrot_geo2proj.gr3 $DATAexec
ln -sf ${PARMnwm}/coastal/${coastal_domain}/hgrid.utm $DATAexec
ln -sf ${PARMnwm}/coastal/${coastal_domain}/hgrid.cpp $DATAexec
ln -sf ${PARMnwm}/coastal/${coastal_domain}/elev.ic $DATAexec
#cpfs ${PARMnwm}/coastal/${coastal_domain}/hgrid.gr3 $DATAexec
#cpfs ${PARMnwm}/coastal/${coastal_domain}/hgrid.ll $DATAexec
#cpfs ${PARMnwm}/coastal/${coastal_domain}/manning.gr3 $DATAexec
#cpfs ${PARMnwm}/coastal/${coastal_domain}/vgrid.in $DATAexec
#cpfs ${PARMnwm}/coastal/${coastal_domain}/bctides.in $DATAexec
#cpfs ${PARMnwm}/coastal/${coastal_domain}/windrot_geo2proj.gr3 $DATAexec
#cpfs ${PARMnwm}/coastal/${coastal_domain}/hgrid.utm $DATAexec
#cpfs ${PARMnwm}/coastal/${coastal_domain}/hgrid.cpp $DATAexec
#cpfs ${PARMnwm}/coastal/${coastal_domain}/elev.ic $DATAexec
# ln -sf %ECF_HOME%/coastal/${coastal_domain}/tvd.prop $DATAexec


if [[ -f ${PARMnwm}/coastal/${coastal_domain}/station.in ]]; then
    ln -sf ${PARMnwm}/coastal/${coastal_domain}/station.in $DATAexec
    #cpfs ${PARMnwm}/coastal/${coastal_domain}/station.in $DATAexec
fi

ln -sf ${PARMnwm}/coastal/${coastal_domain}/element_areas.txt $DATAexec

mkdir -p $DATAexec/sflux
for f in ${PARMnwm}/coastal/${coastal_domain}/sflux/*; do
  cp $f $DATAexec/sflux/
done

#cp --no-preserve=mode -r ${PARMnwm}/coastal/${coastal_domain}/sflux $DATAexec
#
#for the elevation correction
if [[ -f ${PARMnwm}/coastal/${coastal_domain}/elevation_correction.csv ]]; then
      ln -sf ${PARMnwm}/coastal/${coastal_domain}/elevation_correction.csv $DATAexec
fi

}
