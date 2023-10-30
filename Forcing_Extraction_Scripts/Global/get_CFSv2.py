# Quick and dirty program to pull down operational 
# CFSv2 forecast data for each ensemble member, for
# each six hour forecast going out to 30 days. 

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
import smtplib
from email.mime.text import MIMEText
import time
import argparse
import pathlib

def main(args):
    outDir = args.outDir
    lookBackHours = args.lookBackHours
    cleanBackHours = args.cleanBackHours
    lagBackHours = args.lagBackHours

    pid = os.getpid()
    lockFile = outDir + "/GET_CFSV2.lock"

    # First check to see if lock file exists, if it does, throw error message as
    # another pull program is running. If lock file not found, create one with PID.
    if os.path.isfile(lockFile):
        fileLock = open(lockFile,'r')
        pid = fileLock.readline()
        print("ERROR: Another CFSv2 Fetch Program Running. PID: " + pid + ". Please remove lockfile before attempting to execute another file extraction. Exiting script")
        sys.exit(1)
    else:
        fileLock = open(lockFile,'w')
        fileLock.write(str(os.getpid()))
        fileLock.close()

    dNowUTC = datetime.datetime.utcnow()
    dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)
    fcstHrsDownload = 768
    ensNum = "01"
    ncepHTTP = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod"

    for hour in range(cleanBackHours,lookBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)
        # Go back in time and clean out any old data to conserve disk space.
        if dCurrent.hour != 0 and dCurrent.hour != 6 and dCurrent.hour != 12 and dCurrent.hour != 18:
            continue # This is not a CFS cycle hour.
        else:
            # Compose path to directory containing data.
            cfsCleanDir = outDir + "/cfs." + dCurrent.strftime('%Y%m%d') + "/" + \
                      dCurrent.strftime('%H') + "/6hrly_grib_" + ensNum
            # Check to see if directory exists. If it does, remove it.
            if os.path.isdir(cfsCleanDir):
                #print("Removing old CFS data from: " + cfsCleanDir)
                shutil.rmtree(cfsCleanDir)
            # If the subdirectory is empty, remove it.
            cfsCleanDir = outDir + "/cfs." + dCurrent.strftime('%Y%m%d') + "/" + \
                      dCurrent.strftime('%H')
            if os.path.isdir(cfsCleanDir):
                if len(os.listdir(cfsCleanDir)) == 0:
                    #print("Removing empty directory: " + cfsCleanDir)
                    shutil.rmtree(cfsCleanDir)
            cfsCleanDir = outDir + "/cfs." + dCurrent.strftime('%Y%m%d')
            if os.path.isdir(cfsCleanDir):
                if len(os.listdir(cfsCleanDir)) == 0:
                    #print("Removing empty directory: " + cfsCleanDir)
                    shutil.rmtree(cfsCleanDir)

    # Now that cleaning is done, download files within the download window. 
    for hour in range(lookBackHours,lagBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        if dCurrent.hour != 0 and dCurrent.hour != 6 and dCurrent.hour != 12 and dCurrent.hour != 18:
            continue # THis is not a GFS cycle hour.
        else:
            cfsOutDir1 = outDir + "/cfs." + dCurrent.strftime('%Y%m%d')
            if not os.path.isdir(cfsOutDir1):
                os.mkdir(cfsOutDir1)

            cfsOutDir2 = outDir + "/cfs." + dCurrent.strftime('%Y%m%d') + "/" + \
                     dCurrent.strftime('%H')
            if not os.path.isdir(cfsOutDir2):
                os.mkdir(cfsOutDir2)

            cfsOutDir = outDir + "/cfs." + dCurrent.strftime('%Y%m%d') + "/" + \
                    dCurrent.strftime('%H') + "/6hrly_grib_" + ensNum

            httpDownloadDir = ncepHTTP + "/cfs." + dCurrent.strftime('%Y%m%d') + "/" + dCurrent.strftime('%H') + "/" + "6hrly_grib_01"

            if not os.path.isdir(cfsOutDir):
                os.mkdir(cfsOutDir)
            # Download hourly files from NCEP to hour 120.
            for hrDownload in range(0,fcstHrsDownload,6):
                dCurrent2 = dCurrent + datetime.timedelta(seconds=3600*hrDownload)
                fileDownload = "flxf" + dCurrent2.strftime('%Y%m%d%H') + \
                       "." + ensNum + "." + dCurrent.strftime('%Y%m%d%H') + ".grb2"
                url = httpDownloadDir + "/" + fileDownload
                outFile = cfsOutDir + "/" + fileDownload
                if not os.path.isfile(outFile):
                    download_complete = False
                    start_time = time.time()
                    timer = 0.0
                    print("Pulling CFSv2 file: " + url)
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
    parser.add_argument('--lookBackHours', type=int, default=24, help="How many hours to look back for forecast data cycles")
    parser.add_argument('--cleanBackHours', type=int, default=720, help="Period between this time and the beginning of the lookback period to cleanout old data")
    parser.add_argument('--lagBackHours', type=int, default=6, help="Wait at least this long back before searching for files")


    return parser.parse_args()

if __name__ == "__main__":
    args = get_options()
    main(args)

