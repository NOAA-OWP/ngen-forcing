#!/usr/bin/env bash 

###############################################################################
#  Program Name: nwmcopy.sh                                                   #
#                                                                             #
#  Author(s)/Contact(s): NWC                                                  #
#                                                                             #
#  copy files to and from the $COM and  $DBN alert directories                #
#                                                                             #
#  Input: directory in $COM                                                   #
#                                                                             #
#  Output: files                                                              #
#                                                                             #
# For non-fatal errors output is witten to $DATA/LOGS                         #
#                                                                             #
# Origination                                                  Jun, 2019      #
#    OWP        Dec, 2021      Port to WCOSS2                                 #
#                                                                             #
###############################################################################
# --------------------------------------------------------------------------- #

#######################################
# nwm_copy and nwm_postcopy utilize 
# multiple cores to speed up file copying
#######################################
function nwm_copy () {
  msg="Begin copying file at `date`"
  postmsg "${pgmout}" "${msg}"

  dir=$1
  fourcyclesago=$($NDATE -4 ${PDY}${cyc})
  echo "#!/usr/bin/env bash" > $DATA/nwmcopyscript
  for file in $COMIN/${dir}/*
  do
     test -f "$file" || continue 
     filedatetime=$(basename $file | cut -c1-4 )$(basename $file | cut -c6-7 )$(basename $file | cut -c9-10)$(basename $file | cut -c12-13 )
     #
     # Copy only files of the past 4 hours 
     #
     if [ $fourcyclesago -le $filedatetime ]; then
       echo "cpfs $file $DATA/$(basename $file)" >> $DATA/nwmcopyscript
     fi
  done

  #
  # Previous day
  #
  if [ ${cyc} -le 4 ]; then
     for file in $COMINm1/${dir}/*
     do
        test -f "$file" || continue 
        filedatetime=$(basename $file | cut -c1-4 )$(basename $file | cut -c6-7 )$(basename $file | cut -c9-10)$(basename $file | cut -c12-13 )
        #
        # Copy only files of the past 4 hours 
        #
        if [ $fourcyclesago -le $filedatetime ]; then
          echo "cpfs $file $DATA/$(basename $file)" >> $DATA/nwmcopyscript
        fi
     done
  fi

  chmod 755 $DATA/nwmcopyscript

  #aprun -j1 -n$((NODES*NCORES)) -N${NCORES} cfp $DATA/nwmcopyscript
  ${CFPCOMMAND} $DATA/nwmcopyscript
  export err=$?; err_chk

  msg="Ending copy file at `date`"
  postmsg "$pgmout" "$msg"

}

function nwm_postcopy () {
  msg="Begin post copying file at `date`"
  postmsg "${pgmout}" "${msg}"

  dir=$1
  suffix=$2
  fourcyclesago=$($NDATE -4 ${PDY}${cyc})
  echo "#!/usr/bin/env bash" > $DATA/nwmpostcopyscript
  for file in $DATA/*.${suffix}
  do
     test -f "$file" || continue 
     filedatetime=$(basename $file | cut -c1-4 )$(basename $file | cut -c6-7 )$(basename $file | cut -c9-10)$(basename $file | cut -c12-13 )
     #
     # Copy only files of the past 4 hours 
     #
     if [ $fourcyclesago -le $filedatetime ]; then
        #
	# NOTE: Here, we assume $COMOUT == $COMIN, otherwise, it doesn't work.
	#
        #export COMOUT_ROOT=${COMOUT_ROOT:-${COMROOT}/${NET}/${envir}}
        export COMOUT_ROOT=${COMOUT_ROOT:-${COMROOT}/${NET}/${nwm_ver}}
        Outdir=${COMOUT_ROOT}/${RUN}.$(basename $file | cut -c1-4 )$(basename \
            $file | cut -c6-7 )$(basename $file | cut -c9-10)/${dir}

        if [ ! -e $Outdir ]; then 
		mkdir -p $Outdir
	fi

	copyandalert=true

	#
	# if the file exists and was not changed, don't copy and alert
	#
	if [ -e ${Outdir}/$(basename $file) ]; then
	    diff ${file} ${Outdir}/$(basename $file) > /dev/null 2>&1 
	    if [ $? -eq 0 ]; then
		    copyandalert=false
            fi
	fi

	if [[ "$copyandalert" == true ]]; then
            echo "cpfs ${file} ${Outdir}/$(basename $file); if [ "$SENDDBN" = YES ]; then $DBNROOT/bin/dbn_alert  MODEL ${DBN_ALERT_TYPE} $job ${Outdir}/$(basename $file); fi" >> $DATA/nwmpostcopyscript
	fi

     fi
  done

  chmod 755 $DATA/nwmpostcopyscript

  #aprun -j1 -n$((NODES*NCORES)) -N${NCORES} cfp $DATA/nwmpostcopyscript
  ${CFPCOMMAND} $DATA/nwmpostcopyscript
  export err=$?; err_chk

  msg="Ending post copy file at `date`"
  postmsg "$pgmout" "$msg"

}
