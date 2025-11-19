#!/bin/bash

###############################################################################
#  File name: run_usace.sh                                                    #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  05/2022                                           #
#                                                                             #
#                                                                             #
#  Description: Run the US SACE Stream Flow scripts                           #
#               Supported by the NWS Water Predication Center                 #
#                                                                             #
#  12/18/2024   OWP     Download json format datafiles instead of xml         #
#                                                                             #
###############################################################################

#module unload python
#module load python
#module list

#log=$DBNROOT/user/usgs_download/parallel_download_master.log
#log=/gpfs/hps3/ptmp/Zhengtao.Cui/usace_download.log
#OUTDIR=/gpfs/hps3/ptmp/Zhengtao.Cui/ace_json_test

SITE_FILE=./site-file.csv
OUTDIR=$DCOMROOT/usace_streamflow

log=$DBNROOT/log/usace_download.log.`date -u +%Y%m%d`
touchfiles=("$OUTDIR")

cd $DBNROOT/user/usace_download

if [ ! -e ${OUTDIR} ]; then mkdir -p ${OUTDIR}; fi

USACE_DOWNLOAD_MASTER=$DBNROOT/user/usace_download/CWMS_download_current.py

DATE=`/bin/date +%H:%M`

RESETLOGAT="05:28"

if [ "$DATE" = "$RESETLOGAT" ]
then
    echo "Message: reset ${USACE_DOWNLOAD_MASTER} log file"
    rm $log
fi

# Check if process is already running with this package
if pgrep -f CWMS_download_current.py > /dev/null 2>&1
then
    echo "Message: ${USACE_DOWNLOAD_MASTER} package is running"
    for f in ${touchfiles[@]}
    do
        timediff=$((`date +%s` - `stat -c "%Y" $f`))
        if [ $timediff -gt $(( 3600 * 2 )) ] #three hours
        then
                echo "Message: touch file $f is older than 3 hours"
                echo "Message: restart CWMS_download_current.py"
                kill -9 `pgrep -f CWMS_download_current.py`
                nohup python3 ${USACE_DOWNLOAD_MASTER} -f json ${SITE_FILE} $OUTDIR >> $log  2>&1 &
                echo "Message: done restart USACE CWMS_download_current.py"
                break
        fi

    done

else
  echo "Message: ${USACE_DOWNLOAD_MASTER} package NOT running"
  nohup python3 $USACE_DOWNLOAD_MASTER -f json ${SITE_FILE} ${OUTDIR} >> $log  2>&1 &
  echo "Message: ${USACE_DOWNLOAD_MASTER} package started"
fi

## Monitoring adjusted to check for non-zero byte files 09/07/2021
f=$(ls -ltr $OUTDIR | tail -1 | awk '{print $9}')
if [ -s $OUTDIR/${f} ];
then
    #file is larger than 0-byte
    most_recent=`ls -ltr $OUTDIR | tail -1 | awk '{print $6" "$7" "$8}'`
    touch -d "$most_recent" $DBNROOT/tmp/usace_streamflow
else
    echo "Files are empty. Check to see if the website is down." >> $log
fi


##
exit
