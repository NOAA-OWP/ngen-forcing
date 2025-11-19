#!/usr/bin/env python

###############################################################################
#  Module name: usgs_iv_retrieval                                             #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  7/12/2017                                         #
#                                                                             #
#  Description:  The main function to download real time stream flow
#                                                                             #
###############################################################################

import os, sys, time, urllib, getopt, copy
from string import *
import datetime
import find_changed_site_for_huc


def main(argv):
   """
       Function to get output directory

       Return: Output directory
   """
   outputdir = ''
   try:
      opts, args = getopt.getopt(argv,"h:o:",["odir="])
   except getopt.GetoptError:
      print( 'usgs_iv_retrieval.py -o <outputdir>' )
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print( 'usgs_iv_retrieval.py -o <outputdir>' )
         sys.exit()
      elif opt in ("-o", "--odir"):
         outputdir = arg
         if not os.path.exists( outputdir ):
             os.makedirs( arg )
  
   print( 'Output dir is "', outputdir )
   return outputdir

def cleanup_dir( path, numberofdays ):
   """
     Delete files older than number of days in a given directory
     Input: path - the given directory
            numberofdays - number of days older
   """
   if not os.path.isdir(path ):
           return

   now = time.time()
   cutoff = now - ( numberofdays * 86400 )
   files = os.listdir( path )
   for onefile in files:
           if os.path.isfile( path + '/' + onefile ):
                   t = os.stat( path + '/' + onefile )
                   c = t.st_mtime
                   if c < cutoff :
                           os.remove( path + '/' + onefile )
   return

def usgs_iv_retrieval( odir, download_id, hucs ):
  """
       The main function to download real time stream flow

       Input: odir - Output directory
              download_id - The unique identifier of the download process 
              hucs - list of HUCs
  """

#
# For the first loop, get all stations that have data upated in the last 15 minutes.
# In the following loops, use the actual time interval between the loops
#
  timeSinceLast = 15
#
# Time stamp when the process starts
#
  URL_start = [ time.time()] * len( hucs )

  firstloop = True
#
# Got to the infinite loop
#
  while True:

     #
     # touch a file and update the time stamp to indicate it is alive 
     #
     with open( odir + '/usgs_iv_retrieval_' + download_id, 'a'):
             os.utime( odir + '/usgs_iv_retrieval_' + download_id, None )
#
# remove files older than two days in the output directory
#
#  NCO is using a cron job to remove old files. So don't delete old file here
#     print 'cleaning up ...'
#     cleanup_dir( odir, 2 )
#

# Initialize the lists for time tracking
#
     counter_start = time.time()
     total_sites = 0

     huc_seq = 0

#
#    Loop through each HUC
#
     for huc in hucs:
        site_noL = []
        if not firstloop:
           timeSinceLast = ( time.time() - URL_start[ huc_seq ] ) / 60
           #
           #If two queries are less than 2 minutes apart, wait
           # two minutes before the next query.
           #
           if timeSinceLast < 2:
                time.sleep( 120 )
                timeSinceLast = ( time.time() - URL_start[ huc_seq ] ) / 60
            
        URL_start[ huc_seq ] = time.time()

        no_of_sites = []
        #
        # Query the USGS server to find the stations that have updated their 
        # real time data.
        #
        find_changed_site_for_huc.find_changed_sites_for_huc( huc, timeSinceLast, odir, site_noL, no_of_sites )

        print( download_id, ': looptime = ', datetime.datetime.now(), ' num. of sites = ', len(site_noL) )

        #
        # Count the total number of sites that have been updated
        #
        if no_of_sites:
          total_sites += no_of_sites[ 0 ] 

        #
        # If there are any stations that are failed during the query such as
        # no responding from the server, wait for 10 seconds and  try again.
        #
        if not site_noL:
          print ( download_id + ": wait 10 seconds and try again!" )
          time.sleep( 10 )
          no_of_sites = []
          find_changed_site_for_huc.find_changed_sites_for_huc( huc, timeSinceLast, odir, site_noL, no_of_sites )


          #
          # The failed sites need to be counted too.
          #
          if no_of_sites:
            total_sites += no_of_sites[ 0 ] 

        #
        # increment the sequence number
        #
        huc_seq += 1

     URL_end = time.time()
     #
     # print total time spent on this loop
     #
     print( download_id, ': looptime = ', datetime.datetime.now(), round( ( URL_end - counter_start ) / 60, 2), ' minutes' )

     #
     # not the first loop
     #
     firstloop = False

def download_for_hucs( odir, hucs, download_id  ):
    """
       The wrapper to call usgs_iv_retrieval

       Input: odir - Output directory
              download_id - The unique identifier of the download process 
              hucs - list of HUCs
    """

    #
    # call the real time retrieval function
    # 
    usgs_iv_retrieval( odir, download_id, hucs )
