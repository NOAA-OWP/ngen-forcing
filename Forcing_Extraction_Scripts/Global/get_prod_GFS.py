# Quick and dirty program to pull down operational 
# GFS data on the Gaussian grid in GRIB2 format. 

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
    ncepHTTP = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

    pid = os.getpid()
    lockFile = outDir + "/GET_GFS_Full.lock"

    # First check to see if lock file exists, if it does, throw error message as
    # another pull program is running. If lock file not found, create one with PID.
    if os.path.isfile(lockFile):
        fileLock = open(lockFile,'r')
        pid = fileLock.readline()
        print("ERROR: Another GFS Fetch Program Running. PID: " + pid + ". Please remove lockfile before attempting to execute another file extraction. Exiting script")
        sys.exit(1)
    else:
        fileLock = open(lockFile,'w')
        fileLock.write(str(os.getpid()))
        fileLock.close()

    for hour in range(cleanBackHours,lookBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        # Go back in time and clean out any old data to conserve disk space. 
        if dCurrent.hour != 0 and dCurrent.hour != 6 and dCurrent.hour != 12 and dCurrent.hour != 18:
            continue # This is not a GFS cycle hour. 
        else:
            # Compose path to directory containing data. 
            gfsCleanDir = outDir + "/gfs." + dCurrent.strftime('%Y%m%d') + "/" + dCurrent.strftime('%H') + "/atmos"

            # Check to see if directory exists. If it does, remove it. 
            if os.path.isdir(gfsCleanDir):
                print("Removing old GFS data from: " + gfsCleanDir)
                shutil.rmtree(gfsCleanDir)

            # Check to see if parent directory is empty.
            gfsCleanDir = outDir + "/gfs." + dCurrent.strftime('%Y%m%d')
            if os.path.isdir(gfsCleanDir):
                if len(os.listdir(gfsCleanDir)) == 0:
                    print("Removing empty directory: " + gfsCleanDir)
                    shutil.rmtree(gfsCleanDir)

    # Now that cleaning is done, download files within the download window. 
    for hour in range(lookBackHours,lagBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        if dCurrent.hour != 0 and dCurrent.hour != 6 and dCurrent.hour != 12 and dCurrent.hour != 18:
            continue # THis is not a GFS cycle hour. 
        else:
            gfsOutDir1 = outDir + "/gfs." + dCurrent.strftime('%Y%m%d')
            if not os.path.isdir(gfsOutDir1):
                print("Making directory: " + gfsOutDir1)
                os.mkdir(gfsOutDir1)

            gfsOutDir2 = gfsOutDir1 + "/" + dCurrent.strftime('%H') + "/atmos"
        
            httpDownloadDir = ncepHTTP + "/gfs." + dCurrent.strftime('%Y%m%d') + "/" + dCurrent.strftime('%H') + "/atmos"
            if not os.path.isdir(gfsOutDir2):
                print('Making directory: ' + gfsOutDir2)
                os.makedirs(gfsOutDir2)
            # Download hourly files from NCEP to hour 120.
            for hrDownload in range(1,121):
                fileDownload = "gfs.t" + dCurrent.strftime('%H') + \
                       "z.sfluxgrbf" + str(hrDownload).zfill(3) + \
                       ".grib2"
                url = httpDownloadDir + "/" + fileDownload
                outFile = gfsOutDir2 + "/" + fileDownload
                if not os.path.isfile(outFile):
                    download_complete = False
                    start_time = time.time()
                    timer = 0.0
                    print("Pulling GFS file: " + url)
                    while(download_complete == False and timer < 600.0):
                        try:
                            request.urlretrieve(url,outFile)
                            download_complete = True
                        except:
                            timer = time.time() - start_time

                    if(download_complete == False):
                        print("Unable to retrieve: " + url)
                        print("Data may not available yet...")
                        continue
            # Download 3-hour files from hour 120 to hour 240.
            for hrDownload in range(123,243,3):
                fileDownload = "gfs.t" + dCurrent.strftime('%H') + \
                       "z.sfluxgrbf" + str(hrDownload).zfill(3) + \
                       ".grib2"
                url = httpDownloadDir + "/" + fileDownload
                outFile = gfsOutDir2 + "/" + fileDownload
                if not os.path.isfile(outFile):
                    download_complete = False
                    start_time = time.time()
                    timer = 0.0
                    print("Pulling GFS file: " + url)
                    while(download_complete == False and timer < 600.0):
                        try:
                            request.urlretrieve(url,outFile)
                            download_complete = True
                        except:
                            timer = time.time() - start_time

                    if(download_complete == False):
                        print("Unable to retrieve: " + url)
                        print("Data may not available yet...")
                        continue
            # Download 12-hour files from hour 240 to hour 384.
            for hrDownload in range(252,396,12):
                fileDownload = "gfs.t" + dCurrent.strftime('%H') + \
                       "z.sfluxgrbf" + str(hrDownload).zfill(3) + \
                       ".grib2"
                url = httpDownloadDir + "/" + fileDownload
                outFile = gfsOutDir2 + "/" + fileDownload
                if not os.path.isfile(outFile):
                    download_complete = False
                    start_time = time.time()
                    timer = 0.0
                    print("Pulling GFS file: " + url)
                    while(download_complete == False and timer < 600.0):
                        try:
                            request.urlretrieve(url,outFile)
                            download_complete = True
                        except:
                            timer = time.time() - start_time

                    if(download_complete == False):
                        print("Unable to retrieve: " + url)
                        print("Data may not available yet...")
                        continue
    # Remove the LOCK file.
    os.remove(lockFile)

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('outDir', type=str, help="Output directory pathway where the NOMADS data will be downloaded to")
    parser.add_argument('--lookBackHours', type=int, default=24, help="How many hours to look back for forecast data cycles")
    parser.add_argument('--cleanBackHours', type=int, default=240, help="Period between this time and the beginning of the lookback period to cleanout old data")
    parser.add_argument('--lagBackHours', type=int, default=6, help="Wait at least this long back before searching for files")


    return parser.parse_args()

if __name__ == "__main__":
    args = get_options()
    main(args)

