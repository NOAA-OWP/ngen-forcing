#!/bin/bash

###############################################################################
#  File name: run_usgs.sh                                                     #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  12/5/2019                                         #
#  Changed python/2.7.14 to default python module, python/3.6.3               # 
#                                                                             #
#  Description: Run the USGS Stream Flow scripts                              #
#               Supported by the NWS Water Predication Center                 #
#                                                                             #
#                                                                             #  
###############################################################################

module unload python
module load python
module list

#cd /lfs/h1/owp/nwm/noscrub/$LOGNAME/test/packages/nwm.v3.1.0/ush/usgs_download/stream_flow_download

#log=$DBNROOT/user/usgs_download/parallel_download_master.log
log=/lfs/h1/owp/ptmp/$LOGNAME/usgs/usgs_download.log

#OUTDIR=$DCOMROOT/${envir}/usgs_streamflow
OUTDIR=/lfs/h1/owp/ptmp/$LOGNAME/usgs

touchfiles=("$OUTDIR/usgs_iv_retrieval_2604" \
            "$OUTDIR/usgs_iv_retrieval_2634" \
            "$OUTDIR/usgs_iv_retrieval_2920")

PARALLEL_DOWNLOAD_MASTER=./parallel_download_master.py

DATE=`/bin/date +%H:%M`

RESETLOGAT="05:28"

# Check if process is already running with this package
if pgrep -f parallel_download_master.py > /dev/null 2>&1
then
    echo "Message: parallel_download_master.py package is running"
    if [ "$DATE" = "$RESETLOGAT" ]
    then
	echo "Message: reset USGS parallel_download_master log file"
	rm $log
    fi

    for f in ${touchfiles[@]}
    do
	timediff=$((`date +%s` - `stat -c "%Y" $f`))
	if [ $timediff -gt $(( 3600 * 2 )) ] #three hours
	then
		echo "Message: touch file $f is older than 3 hours"
		echo "Message: restart USGS parallel_download_master.py"
                kill -9 `pgrep -f parallel_download_master.py`
		nohup $PARALLEL_DOWNLOAD_MASTER -o $OUTDIR >> $log  2>&1 &
		echo "Message: done restart USGS parallel_download_master.py"
		break
	fi

    done

else
  echo "Message: parallel_download_master.py package NOT running"
  nohup $PARALLEL_DOWNLOAD_MASTER -o $OUTDIR >> $log  2>&1 &
fi
exit
