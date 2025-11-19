#!/bin/bash

# Run the Canadian Stream Flow scripts
# Developed by Tim Hunter at NOAA/GLERL, shamelessly plagiarizing work by 
# Zhengtao Cui at NOAA/OHD

log=/gpfs/hps3/ptmp/Zhengtao.Cui/CanDA/can_download_log.txt
OUTDIR=/gpfs/hps3/ptmp/Zhengtao.Cui/CanDA/test1/

touchfiles=("$OUTDIR")

cd /gpfs/hps3/nwc/noscrub/Zhengtao.Cui/nwtest3/nwm.v2.1/ush/canada_download
PARALLEL_DM_CAN=/gpfs/hps3/nwc/noscrub/Zhengtao.Cui/nwtest3/nwm.v2.1/ush/canada_download/parallel_dm_can.py

DATE=`/bin/date +%H:%M`

RESETLOGAT="05:28"

# Check if process is already running with this package
if pgrep -f parallel_dm_can.py > /dev/null 2>&1
then
   echo "Message: parallel_dm_can.py package is running"
   if [ "$DATE" = "$RESETLOGAT" ]
   then
      echo "Message: reset Canadian parallel_dm_can log file"
      rm $log
   fi

   for f in ${touchfiles[@]}
   do
      timediff=$((`date +%s` - `stat -c "%Y" $f`))
      if [ $timediff -gt $(( 3600 * 3 )) ] #three hours
      then
         echo "Message: touch file $f is older than 3 hours"
         echo "Message: restart Canadian parallel_dm_can.py"
         kill -9 `pgrep -f parallel_dm_can.py`
         nohup $PARALLEL_DM_CAN -o $OUTDIR >> $log  2>&1 &
         echo "Message: done restart Canadian parallel_dm_can.py"
         break
      fi

   done
else
   echo "Message: parallel_dm_can.py package NOT running"
   nohup $PARALLEL_DM_CAN -o $OUTDIR >> $log  2>&1 &
fi
exit
