# Quick and dirty program to pull down operational 
# conus MRMS Radar Quality Index data. 

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
import time
import argparse

def main(args):
    outDir = args.outDir
    lookBackHours = args.lookBackHours
    cleanBackHours = args.cleanBackHours
    lagBackHours = args.lagBackHours

    dNowUTC = datetime.datetime.utcnow()
    dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)
    ncepHTTP = "https://mrms.ncep.noaa.gov/data/2D/MultiSensor_QPE_01H_"

    pid = os.getpid()
    lockFile = outDir + "/GET_MRMS_MultiSensor_CONUS.lock"

    # First check to see if lock file exists, if it does, throw error message as
    # another pull program is running. If lock file not found, create one with PID.
    if os.path.isfile(lockFile):
        fileLock = open(lockFile,'r')
        pid = fileLock.readline()
        print("ERROR: Another MRMS Multisensor CONUS Fetch Program Running. PID: " + pid + ". Please remove lockfile before attempting to execute another file extraction. Exiting script")
        sys.exit(1)
    else:
        fileLock = open(lockFile,'w')
        fileLock.write(str(os.getpid()))
        fileLock.close()

    for hour in range(cleanBackHours, lookBackHours, -1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600 * hour)

        # Compose path to MRMS file to clean for Pass #1
        fileClean = outDir + "/MultiSensor_QPE_01H_Pass1/" + "/MRMS_MultiSensor_QPE_01H_Pass1_00.00_" + dCurrent.strftime('%Y%m%d') + \
            "-" + dCurrent.strftime('%H') + '0000.grib2.gz'

        if os.path.isfile(fileClean):
            print("Removing old file: " + fileClean)
            os.remove(fileClean)

        # Compose path to MRMS file to clean for Pass #2
        fileClean = outDir + "/MultiSensor_QPE_01H_Pass2/" + "/MRMS_MultiSensor_QPE_01H_Pass2_00.00_" + dCurrent.strftime('%Y%m%d') + \
            "-" + dCurrent.strftime('%H') + '0000.grib2.gz'

        if os.path.isfile(fileClean):
            print("Removing old file: " + fileClean)
            os.remove(fileClean)




    for hour in range(lookBackHours,lagBackHours,-1):
        dCycle = dNow - datetime.timedelta(seconds=3600*hour)
        print("Current Step = " + dCycle.strftime('%Y-%m-%d %H'))

        pass1_dir = outDir + "/MultiSensor_QPE_01H_Pass1/" + dCycle.strftime('%Y%m%d')
        if(os.path.isdir(pass1_dir) == False):
            os.makedirs(pass1_dir)

        fileDownload = "MRMS_MultiSensor_QPE_01H_Pass1_00.00_" + dCycle.strftime('%Y%m%d') + \
               "-" + dCycle.strftime('%H') + '0000.grib2.gz'
        url = ncepHTTP + "Pass1/" + fileDownload
        outFile = pass1_dir + "/" + fileDownload
        if os.path.isfile(outFile):
            print(outFile + " already exists in output directory. Continuing")
        else:
            download_complete = False
            start_time = time.time()
            timer = 0.0
            print("Pulling MRMS MultiSensor CONUS Pass #1 file: " + url)
            while(download_complete == False and timer < 120.0):
                try:
                    request.urlretrieve(url,outFile)
                    download_complete = True
                except:
                    timer = time.time() - start_time

            if(download_complete == False):
                print("Unable to retrieve: " + url)
                print("Data may not available yet...")

        pass2_dir = outDir + "/MultiSensor_QPE_01H_Pass2/" + dCycle.strftime('%Y%m%d')
        if(os.path.isdir(pass2_dir) == False):
            os.makedirs(pass2_dir)

        fileDownload = "MRMS_MultiSensor_QPE_01H_Pass2_00.00_" + dCycle.strftime('%Y%m%d') + \
               "-" + dCycle.strftime('%H') + '0000.grib2.gz'
        url = ncepHTTP + "Pass2/" + fileDownload
        outFile = pass2_dir + "/" + fileDownload
        if os.path.isfile(outFile):
            continue
        download_complete = False
        start_time = time.time()
        timer = 0.0
        print("Pulling MRMS MultiSensor CONUS Pass #2 file: " + url)
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
    parser.add_argument('--lookBackHours', type=int, default=25, help="How many hours to look back for forecast data cycles")
    parser.add_argument('--cleanBackHours', type=int, default=240, help="Period between this time and the beginning of the lookback period to cleanout old data")
    parser.add_argument('--lagBackHours', type=int, default=0, help="Wait at least this long back before searching for files")


    return parser.parse_args()

if __name__ == "__main__":
    args = get_options()
    main(args)

