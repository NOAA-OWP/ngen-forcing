#!/usr/bin/env bash 

###############################################################################
#  Program Name: nwm_usgs_timeslices                                          #
#                                                                             #
#  Author(s)/Contact(s): NWC                                                  #
#                                                                             #
#  This is regriding USGS time slices creation                                #
#                                                                             #
#  Input: /lfs/h1/ops/prod/dcom/usgs_streamflow                               #
#                                                                             #
#  Output: YYYY-MM-DD_HH:mm:00.15min.usgsTimeSlice.ncdf                       #
#                                                                             #
# For non-fatal errors output is witten to $DATA/LOGS                         #
#                                                                             #
# Origination                                                  Feb, 2016      #
# OWP           12/07/2021      Port to WCOSS2 system                         #
# OWP           06/18/2024      Change Water ML 2.0 to Json data format       #
#                                                                             #
###############################################################################
# --------------------------------------------------------------------------- #
seton='-xa'
setoff='+xa'

postmsg "$pgmout" "HAS BEGUN on `hostname`"
msg="Starting USGS real time time slices creation at `date`"
postmsg "$pgmout" "$msg"

source $USHnwm/usgs_download/analysis/nwmcopy.sh

set $setoff
echo ' '
echo '********************************'
echo '** NWM USGS TIME SLICES **'
echo '********************************'
echo -e "\nStarting $0 at : `date`\n"
set $seton

if [ ! -e $COMOUT/usgs_timeslices ]; then
  mkdir -p $COMOUT/usgs_timeslices
fi

#################################
# copy raw data files to local working dir
#################################
if [ ! -e $DATA/rawdatafiles ]; then
  mkdir -p $DATA/rawdatafiles
fi

for file in $DCOM/*.json
do
     test -f "$file" || continue 
     cpfs $file $DATA/rawdatafiles
done
#################################
# copy existing data files
#################################
nwm_copy usgs_timeslices

postmsg "$pgmout" "Execute $USHnwm/usgs_download/analysis/make_time_slice_from_usgs_waterml.py"
postmsg "$pgmout" "$USHnwm/usgs_download/analysis/make_time_slice_from_usgs_waterml.py depends on:"
postmsg "$pgmout" "$USHnwm/usgs_download/analysis/TimeSlice.py and $USHnwm/usgs_download/analysis/USGS_Observation.py"

python $USHnwm/usgs_download/analysis/make_time_slice_from_usgs_waterml.py   \
           -i $DATA/rawdatafiles -o $DATA >> $pgmout 2>&1

#$USHnwm/usgs_download/analysis/make_time_slice_from_usgs_waterml.py   \
#           -i $DCOM -o $DATA >> $pgmout 2>&1

export err=$? #; err_chk

if [ "$err" -ne 0 ]; then
   errMsg=$(sed -n 's/^.* - __main__ - ERROR - //p' $pgmout)

   if [ ! -z "$errMsg" ]; then
       echo -e "\n${jobid} failed because:\n\t${errMsg}\n" >> emailMsg
   fi
     
   cat emailMsg | mail.py -v -s "ERROR: NWM USGS timeslice job Failed" $maillist2
else
   errMsg=$(sed -n 's/^[0-2][0-9]\{3\}-[01][0-9]-[0-3][0-9] [0-9:,]* - __main__ - WARNING - \(Input directory\)/\1/p' $pgmout)

   if [ "$errMsg" == "Input directory ${DATA}/rawdatafiles has no USGS Json files or the files are empty!" ]; then
       hour=$(date +"%H"); min=$(date +"%M")
       DISABLE_EMAIL=${DISABLE_USGS_EMAIL_ENV:-NO}
       #
       # Only send the warning message every other hour between minutes 10 to 
       #  25
       #
       if [[ "$DISABLE_EMAIL" == "NO" && $((10#$hour % 2)) == 0 && \
                              "$min" -ge 10 && "$min" -lt 25 ]]; then
          errMsg="Input directory ${DCOM} has no USGS Json files or the files are empty!"
          echo -e "\n${jobid} WARNING:\n\t${errMsg}\n" >> emailMsg
          cat emailMsg | mail.py -v -s "WARNING: NWM USGS timeslice job has empty input" $maillist2
       else
          postmsg "$pgmout" "WARNING: $errMsg"
       fi
   else
       err_chk

       #
       # send output time slice files to /com
       #
       if [ "$SENDCOM" == "YES" ]; then
         nwm_postcopy usgs_timeslices usgsTimeSlice.ncdf
       fi
   fi
fi

export err=$?; err_chk

postmsg "$pgmout" "Finishing USGS real time time slices creation on `date`."

#-----------------------------------
# Save log file to $COMOUT/logs
#----------------------------------
if [ ! -e $COMOUT/logs ]; then
  mkdir -p $COMOUT/logs
fi

if [ -e $DATAlogs ]; then 
    gzip $pgmout 
    cp -rf ${pgmout}.gz $COMOUT/logs 
fi

export err=$?; err_chk

set $setoff
echo ' '
echo -e "\nEnding at : `date`\n"
echo '***********************************'
