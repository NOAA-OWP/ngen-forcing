# Quick and dirty program to pull down operational 
# conus HRRR data (surface files).

# Logan Karsten
# National Center for Atmospheric Research
# Research Applications Laboratory

import datetime
import urllib
from urllib import request
import http
from http import cookiejar
import os
import sys
import shutil
import time
import argparse

def main(args):
    outDir = args.outDir
    lookBackHours = args.lookBackHours
    cleanBackHours = args.cleanBackHours
    lagBackHours = args.lagBackHours

    dNowUTC = datetime.datetime.utcnow()
    dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)
    ncepHTTP = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/v4.1"

    pid = os.getpid()
    lockFile = outDir + "/GET_Conus_StageIV.lock"

    # First check to see if lock file exists, if it does, throw error message as
    # another pull program is running. If lock file not found, create one with PID.
    if os.path.isfile(lockFile):
        fileLock = open(lockFile,'r')
        pid = fileLock.readline()
        print("ERROR: Another CONUS StageIV Fetch Program Running. PID: " + pid + ". Please remove lockfile before attempting to execute another file extraction. Exiting script")
        sys.exit(1)
    else:
        fileLock = open(lockFile,'w')
        fileLock.write(str(os.getpid()))
        fileLock.close()

    for hour in range(cleanBackHours,lagBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        # Compose path to directory containing data.
        pcpanlCleanDir = outDir + "/pcpanl." + dCurrent.strftime('%Y%m%d')

        # Check to see if directory exists. If it does, remove it. 
        if os.path.isdir(pcpanlCleanDir):
            print("Removing old CONUS StageIV data from: " + pcpanlCleanDir)
            shutil.rmtree(pcpanlCleanDir)

    # Now that cleaning is done, download files within the download window. 
    for hour in range(lookBackHours,lagBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        pcpanlOutDir = outDir #+ "/pcpanl." + dCurrent.strftime('%Y%m%d')
        if not os.path.isdir(pcpanlOutDir):
            os.mkdir(pcpanlOutDir)

        httpDownloadDir = ncepHTTP + "/pcpanl." + dCurrent.strftime('%Y%m%d') 
        fileDownload = "st4_conus." + dCurrent.strftime('%Y%m%d%H') + ".01h.grb2"
        url = httpDownloadDir + "/" + fileDownload
        outFile = pcpanlOutDir + "/" + fileDownload
        if not os.path.isfile(outFile):
            download_complete = False
            start_time = time.time()
            timer = 0.0
            print("Pulling CONUS StageIV file: " + url)
            while(download_complete == False and timer < 120.0):
                try:
                    request.urlretrieve(url,outFile)
                    download_complete = True
                except:
                    timer = time.time() - start_time

        if(download_complete == False):
            print("Unable to retrieve: " + url)
            print("Data may not available yet...")

    # Remove the LOCK file.
    os.remove(lockFile)

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('outDir', type=str, help="Output directory pathway where the NOMADS data will be downloaded to")
    parser.add_argument('--lookBackHours', type=int, default=36, help="How many hours to look back for forecast data cycles")
    parser.add_argument('--cleanBackHours', type=int, default=240, help="Period between this time and the beginning of the lookback period to cleanout old data")
    parser.add_argument('--lagBackHours', type=int, default=0, help="Wait at least this long back before searching for files")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_options()
    main(args)

