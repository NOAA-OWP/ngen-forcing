# Quick and dirty program to pull down operational 
# conus Rapid Refresh data (surface files).

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
    ncepHTTP = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod"

    pid = os.getpid()
    lockFile = outDir + "/GET_Conus_RAP.lock"

    # First check to see if lock file exists, if it does, throw error message as
    # another pull program is running. If lock file not found, create one with PID.
    if os.path.isfile(lockFile):
        fileLock = open(lockFile,'r')
        pid = fileLock.readline()
        print("ERROR: Another CONUS RAP Fetch Program Running. PID: " + pid + ". Please remove lockfile before attempting to execute another file extraction. Exiting script")
        sys.exit(1)
    else:
        fileLock = open(lockFile,'w')
        fileLock.write(str(os.getpid()))
        fileLock.close()

    for hour in range(cleanBackHours,lagBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        # Compose path to directory containing data.
        rapCleanDir = outDir + "/rap." + dCurrent.strftime('%Y%m%d')

        # Check to see if directory exists. If it does, remove it. 
        if os.path.isdir(rapCleanDir):
            print("Removing old CONUS RAP data from: " + rapCleanDir)
            shutil.rmtree(rapCleanDir)

    # Now that cleaning is done, download files within the download window. 
    for hour in range(lookBackHours,lagBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        rapOutDir = outDir + "/rap." + dCurrent.strftime('%Y%m%d')
        if not os.path.isdir(rapOutDir):
            os.mkdir(rapOutDir)

        if dCurrent.hour == 3 or dCurrent.hour == 9 or dCurrent.hour == 15 or dCurrent.hour == 21:
            # RAP cycles every six hours produce forecasts out to 39 hours.
            nFcstHrs = 39
        else:
            # Otherwise, 21 hour forecasts. 
            nFcstHrs = 21

        for hrDownload in range(0,nFcstHrs+1):
            httpDownloadDir = ncepHTTP + "/rap." + dCurrent.strftime('%Y%m%d')
            fileDownload = "rap.t" + dCurrent.strftime('%H') + \
                       "z.awp130bgrbf" + str(hrDownload).zfill(2) + ".grib2"
            url = httpDownloadDir + "/" + fileDownload
            outFile = rapOutDir + "/" + fileDownload
            if not os.path.isfile(outFile):
                download_complete = False
                start_time = time.time()
                timer = 0.0
                print("Pulling CONUS RAP file: " + url)
                while(download_complete == False and timer < 600.0):
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
    parser.add_argument('--lookBackHours', type=int, default=30, help="How many hours to look back for forecast data cycles")
    parser.add_argument('--cleanBackHours', type=int, default=240, help="Period between this time and the beginning of the lookback period to cleanout old data")
    parser.add_argument('--lagBackHours', type=int, default=1, help="Wait at least this long back before searching for files")


    return parser.parse_args()

if __name__ == "__main__":
    args = get_options()
    main(args)
