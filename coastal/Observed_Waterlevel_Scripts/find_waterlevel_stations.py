#!/usr/bin/env python

###############################################################################
#  Module name: find_waterlevel_stations                                      #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com                            #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  11/12/2024                                        #
#                                                                             #
#  Description: Download a list of waterlevel stations                     #
#                                                                             #
###############################################################################

import os, sys, time, urllib.request, getopt, copy
from string import *
import datetime
import json
#import fetch_sites

#
#

def find_waterlevel_stations( site_noL ):
        """
        Find all of the the water level stations

        Output: site_noL - list of site numbers
        """

        #
        # Consturct an url for the CO-OPS Metadata AP  server
        #
        #URL = ('https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=waterlevels' ) 
        URL = ('https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=historicwl' ) 

        print( ' URL = ', URL )

        #
        # Connect to the server
        #
        try:
           rno    = urllib.request.urlopen(URL)
        except IOError as e:
                print ( 'Failed connecting to the server. No data was downloaded!' )
 #                print ( 'Reason: ', e.reason )
                return

        #
        # Download the station list 
        #
        try:
           json_data = json.loads(rno.read().decode('utf-8'))
        except IOError as e:
                print( datetime.datetime.now(), end = " --- " )
                print( 'Failed reading the server. No data was downloaded!' )
 #                print ( 'Reason: ', e.reason )
                return

        #
        # Close the connection
        #
        rno.close()
            
        print( json_data['count'] ) 
        for sta in  json_data['stations']:
            site_noL.append( sta[ 'id' ] )

        return
