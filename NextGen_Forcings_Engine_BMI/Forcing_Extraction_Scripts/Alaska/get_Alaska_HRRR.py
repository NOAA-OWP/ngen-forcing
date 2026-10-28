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

    #dtProc = endDate - begDate
    #nCycles = dtProc.days*24 + dtProc.seconds/3600.0
    ncepHTTP = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"

    pid = os.getpid()
    lockFile = outDir + "/GET_Alaska_HRRR.lock"

    # First check to see if lock file exists, if it does, throw error message as
    # another pull program is running. If lock file not found, create one with PID.
    if os.path.isfile(lockFile):
        fileLock = open(lockFile,'r')
        pid = fileLock.readline()
        print("ERROR: Another Alaska HRRR Fetch Program Running. PID: " + pid + ". Please remove lockfile before attempting to execute another file extraction. Exiting script")
        sys.exit(1)
    else:
        fileLock = open(lockFile,'w')
        fileLock.write(str(os.getpid()))
        fileLock.close()

    for hour in range(cleanBackHours,lagBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        # Compose path to directory containing data.
        hrrrCleanDir = outDir + "/hrrr." + dCurrent.strftime('%Y%m%d')  + "/alaska"

        # Check to see if directory exists. If it does, remove it. 
        if os.path.isdir(hrrrCleanDir):
            print("Removing old HRRR data from: " + hrrrCleanDir)
            shutil.rmtree(hrrrCleanDir)

    # Now that cleaning is done, download files within the download window. 
    for hour in range(lookBackHours,lagBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        hrrrOutDir = outDir + "/hrrr." + dCurrent.strftime('%Y%m%d')  + "/alaska"
        if(os.path.isdir(hrrrOutDir) == False):
            os.makedirs(hrrrOutDir)

        # HRRR cycles every 3 hours
        if dCurrent.hour % 3 == 0:

            if dCurrent.hour == 0:
                # HRRR 00z cycle products forecasts out to 48 hours.
                nFcstHrs = 48
            else:
                # Otherwise, 18 hour forecasts. 
                nFcstHrs = 18

        for hrDownload in range(0,nFcstHrs+1):
            httpDownloadDir = ncepHTTP + "/hrrr." + dCurrent.strftime('%Y%m%d') + "/alaska"
            fileDownload = "hrrr.t" + dCurrent.strftime('%H') + \
                   "z.wrfsfcf" + str(hrDownload).zfill(2) + ".ak.grib2"
            url = httpDownloadDir + "/" + fileDownload
            outFile = hrrrOutDir + "/" + fileDownload
            if not os.path.isfile(outFile):
                download_complete = False
                start_time = time.time()
                timer = 0.0
                print("Pulling Alaska HRRR file: " + url)
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
    parser.add_argument('--lookBackHours', type=int, default=36, help="How many hours to look back for forecast data cycles")
    parser.add_argument('--cleanBackHours', type=int, default=240, help="Period between this time and the beginning of the lookback period to cleanout old data")
    parser.add_argument('--lagBackHours', type=int, default=1, help="Wait at least this long back before searching for files")


    return parser.parse_args()

if __name__ == "__main__":
    args = get_options()
    main(args)


