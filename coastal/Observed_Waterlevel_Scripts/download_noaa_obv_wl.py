#!/usr/bin/env python

###############################################################################
#  File name: download_noaa_obv_wl.py                                         #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  11/13/2024                                        #
#                                                                             #
#  Description: Download the NOAA observed water level data from the CO-OPS   #
#               server                                                        #
#                                                                             #
###############################################################################

"""
   Syntax to run:
      download_noaa_obv_wl.py -o <outputdir> -b <begin_date> -e <end_date> -m 
   where -o defines the ouput directory
         -b defines the begin time for the timeseries
         -s defines the end time for the timeseries
         -m is optional. If not given, hourly height will be downloaded, 
            otherwise, 6 minute inteval water leel data will be downloaded.

   Accepted formats for begin and end time include:
    yyyyMMdd
    yyyyMMdd HH:mm
    MM/dd/yyyy
    MM/dd/yyyy HH:mm

   Example execution:
     python download_noaa_obv_wl.py -o "./testdir" -b 20241101 -e 20241112 -m
     or
     python download_noaa_obv_wl.py -o "./testdir" -b 20241101 -e 20241112

"""


import os, sys, time, urllib.request, getopt
import datetime
import json
import find_waterlevel_stations

if __name__ == "__main__":
    odir=''
    starttime=None
    endtime=None
    ishourly=True
    site_noL = []
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ho:b:e:l:m",["odir=", "begin=", "end=", "list="])
    except getopt.GetoptError:
      print('download_noaa_obv_wl.py -o <outputdir> -b <begin_date> -e <end_date> -m') 
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print(   \
           'download_noaa_obv_wl.py -o <outputdir> -s <start_date> -e <end_date> -l <station_list> -m') 
         sys.exit()
      elif opt in ('-o', "--odir" ):
         odir = arg
         if not os.path.exists( odir ):
                 raise RuntimeError( 'FATAL ERROR: outputdir ' + \
                                 odir + ' does not exist!' )
      elif opt in ('-b', "--begin" ):
         starttime = arg
      elif opt in ('-e', "--end" ):
         endtime = arg
      elif opt in ('-l', "--list" ):
         site_noL = arg.split(',')
      elif opt in ('-m' ):
         ishourly = False
 
    if not starttime:
         raise RuntimeError( 'FATAL ERROR: starttime is not defined!' )

    if not endtime:
         raise RuntimeError( 'FATAL ERROR: endtime is not defined!' )
  
    if not site_noL:
      find_waterlevel_stations.find_waterlevel_stations(site_noL)

    failed_sites = []
    for sta in site_noL:
        print( sta )
        #URL="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_level&application=NOS.COOPS.TAC.WL&begin_date=20241101&end_date=20241112&datum=MLLW&station="+sta+"&time_zone=GMT&units=metric&format=json"
        if ishourly:
           URL="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=hourly_height&application=NOS.COOPS.TAC.WL&begin_date="+starttime+"&end_date="+endtime+"&datum=MLLW&station=" + sta + "&time_zone=GMT&units=metric&format=json"
        else:
           URL="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_level&application=NOS.COOPS.TAC.WL&begin_date="+starttime+"&end_date="+endtime+"&datum=MLLW&station=" + sta + "&time_zone=GMT&units=metric&format=json"
        print( "URL=", URL )

        # Connect to the server
        # 
        try:
            rno = urllib.request.urlopen(URL)
        except IOError as e:
            print( 'WARNING: site : ', sta, ' skipped - ' + e.reason ) #, e.reason 
            #
            # If failed, remember the station no and continue
            # 
            failed_sites.append( sta )
            continue

        #
        # Write the water level data to json files
        #
        jso = open(odir+'/'+sta+'.json','w')

        try:
             json_data = json.loads(rno.read().decode('utf-8'))
             json.dump(json_data, jso, indent=2)
              
             print( datetime.datetime.now(), end = " --- " )
             print( 'Successfully downloaded water level data for  station: ', sta)
        except IOError as e:
            print( datetime.datetime.now(), end = " --- " )
            print( 'WARNING: station : ', sta, ' skipped - ', e.reason()) 
            failed_sites.append( sta )
            continue

        #
        # Close the connection and json files
        #
        rno.close()
        jso.close()

        #Download the datum

        URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/" + sta + "/datums.json?units=metric"
        # Connect to the server
        # 
        try:
            rno = urllib.request.urlopen(URL)
        except IOError as e:
            print( 'WARNING: site : ', sta, ' skipped - ' + e.reason ) #, e.reason 
            #
            # If failed, remember the station no and continue
            # 
            failed_sites.append( sta )
            continue

        #
        # Write the water level data to json files
        #
        jso = open(odir+'/'+sta+'_datum.json','w')

        try:
             json_data = json.loads(rno.read().decode('utf-8'))
             json.dump(json_data, jso, indent=2)
              
             print( datetime.datetime.now(), end = " --- " )
             print( 'Successfully downloaded datum for  station: ', sta)
        except IOError as e:
            print( datetime.datetime.now(), end = " --- " )
            print( 'WARNING: station : ', sta, ' skipped - ', e.reason()) 
            failed_sites.append( sta )
            continue
        #
        # Close the connection and json files
        #
        rno.close()
        jso.close()
