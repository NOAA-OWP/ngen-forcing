# Quick and dirty program to pull down operational 
# NBM data on the Gaussian grid in GRIB2 format. 

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
import requests
from bs4 import BeautifulSoup
import argparse

def get_url_paths(url, ext='', params={}):
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return parent

def main(args):
    outDir = args.outDir
    lookBackHours = args.lookBackHours
    cleanBackHours = args.cleanBackHours
    lagBackHours = args.lagBackHours

    dNowUTC = datetime.datetime.utcnow()
    dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)
    ncepHTTP = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/v4.2"

    pid = os.getpid()
    lockFile = outDir + "/GET_NBM_Full.lock"

    # First check to see if lock file exists, if it does, throw error message as
    # another pull program is running. If lock file not found, create one with PID.
    if os.path.isfile(lockFile):
        fileLock = open(lockFile,'r')
        pid = fileLock.readline()
        warningMsg =  "WARNING: Another NBM Global FV3 Fetch Program Running. PID: " + pid
        warningOut(warningMsg,warningTitle,emailAddy,lockFile)
    else:
        fileLock = open(lockFile,'w')
        fileLock.write(str(os.getpid()))
        fileLock.close()

    for hour in range(cleanBackHours,lookBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        # Compose path to directory containing data. 
        nbmCleanDir = outDir + "/blend." + dCurrent.strftime('%Y%m%d') + "/" + dCurrent.strftime('%H') + "/core"

        # Check to see if directory exists. If it does, remove it. 
        if os.path.isdir(nbmCleanDir):
            print("Removing old NBM data from: " + nbmCleanDir)
            shutil.rmtree(nbmCleanDir)

        # Check to see if parent directory is empty.
        nbmCleanDir = outDir + "/blend." + dCurrent.strftime('%Y%m%d')
        if os.path.isdir(nbmCleanDir):
            if len(os.listdir(nbmCleanDir)) == 0:
                print("Removing empty directory: " + nbmCleanDir)
                shutil.rmtree(nbmCleanDir)


    # Now that cleaning is done, download files within the download window. 
    for hour in range(lookBackHours,lagBackHours,-1):
        # Calculate current hour.
        dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

        nbmOutDir1 = outDir + "/blend." + dCurrent.strftime('%Y%m%d')
        if not os.path.isdir(nbmOutDir1):
            print("Making directory: " + nbmOutDir1)
            os.mkdir(nbmOutDir1)

        nbmOutDir2 = nbmOutDir1 + "/" + dCurrent.strftime('%H') + "/core"
        
        httpDownloadDir = ncepHTTP + "/blend." + dCurrent.strftime('%Y%m%d') + "/" + dCurrent.strftime('%H') + "/core/"
        if not os.path.isdir(nbmOutDir2):
            print('Making directory: ' + nbmOutDir2)
            os.makedirs(nbmOutDir2)

        # Request list of NBM CONUS files in directory since
        # their forecast output intervals are inconsistent
        ext = ".co.grib2"
        nbm_urls = get_url_paths(httpDownloadDir,ext)
        for i in range(len(nbm_urls)):
            fileDownload = nbm_urls[i].split('/')[-1]
            outFile = nbmOutDir2 + "/" + fileDownload
            if not os.path.isfile(outFile):
                download_complete = False
                start_time = time.time()
                timer = 0.0
                print("Pulling NBM CONUS file: " + nbm_urls[i])
                while(download_complete == False and timer < 600.0):
                    try:
                        request.urlretrieve(nbm_urls[i],outFile)
                        download_complete = True
                    except:
                        timer = time.time() - start_time

            if(download_complete == False):
                print("Unable to retrieve: " + nbm_urls[i])
                print("Data may not available yet...")

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
