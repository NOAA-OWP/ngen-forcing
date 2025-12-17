#!/usr/bin/env python

###############################################################################
#  Module name: fetch_sites                                                   #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  7/12/2017                                         #
#                                                                             #
#  Description: Download real time stream flow data from the USGS server for  #
#               a given list of stations.                                     #
#                                                                             #
#  06/11/2024 ChamP   Replaced "waterml,2.0" with "json" format in queries to #
#                     the USGS data service.                                  #
###############################################################################

import os, sys, time, urllib.request, getopt
from string import *
import datetime
import json

def fetch_sites( site_nos, odir, failed_sites ):
    """
       Download real time stream flow data from the USGS server for 
       a given list of stations.

       Input: site_nos - list of USGS station numbers
              odir - the output directory
              failed_sites: list of USGS station numbers that failed to 
                            download
    """

    #
    # Loop through each station
    #
    for site_no in site_nos:
          #
          # Construct the query URL
          # 
          #URL = ( 'https://staging.waterservices.usgs.gov/nwis/iv/?sites='+
          URL = ( 'https://waterservices.usgs.gov/nwis/iv/?sites='+
                site_no+
                '&format=json&parameterCd=00060&period=PT6H' )
          #'&format=waterml,2.0&parameterCd=00060&period=PT6H' )
          #
          # Connect to the server
          # 
          try:
            rno = urllib.request.urlopen(URL)
          except IOError as e:
            print( 'WARNING: site : ', site_no, ' skipped - ') #, e.reason 
            #
            # If failed, remember the station no and continue
            # 
            failed_sites.append( site_no )
            continue

          #
          # Write the real time flow data to waterML files
          #
          jso = open(odir+'/'+site_no+'.json','w')

          try:
             json_data = json.loads(rno.read().decode('utf-8'))
             json.dump(json_data, jso, indent=2)
              
             with open( odir + '/fetch_data_last_success', 'a'):
                  os.utime( odir + '/fetch_data_last_success', None )
             print( datetime.datetime.now(), end = " --- " )
             print( 'Successfully downloaded stream flow data for  station: ', site_no)
          except IOError as e:
            print( datetime.datetime.now(), end = " --- " )
            print( 'WARNING: station : ', site_no, ' skipped - ') #, e.reason 
            failed_sites.append( site_no )
            continue

          #
          # Close the connection and WaterML files
          #
          rno.close()
          jso.close()
    return
