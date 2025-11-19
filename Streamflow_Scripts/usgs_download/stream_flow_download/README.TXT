1) To start the download processes, at the prompt,

./run_usgs.sh

or put this script in a cron job. It will clear the log file every day at 5:28. 

Output is saved in the directory pointed to by the $OUTDIR variable. You can change the output directory by redefining the $OUTDIR variable in run_usgs.sh

The log file is saved to the file defined by the $log variable. You can change the log file by modifying the $log variable in the run_usgs.sh.

The log file is reset daily at 5:28. To change the time, redefine the $RESETLOGAT variable.

The empty file, 'fetch_data_last_success', in the output directory is touched at every successful download. It can be used to monitor the program.

- To kill run_usgs.sh
ps -ef|grep parallel_download_master.py
kill -9 `pgrep -f parallel_download_master.py`

- Here is the URL to download usgs stream flow data
SITE_NO=05018000

#site URL = https://waterservices.usgs.gov/nwis/iv/?sites=05018000&format=waterml,2.0&parameterCd=00060&period=PT6H
site URL = https://waterservices.usgs.gov/nwis/iv/?sites=05018000&format=json&parameterCd=00060&period=PT6H

2024-04-30 17:29:55.757442 --- Successfully downloaded stream flow data for  station:  05018000
