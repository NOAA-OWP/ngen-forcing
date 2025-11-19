#! /usr/bin/env python
###############################################################################
#  File name: make_time_slice_from_ace.py                                     #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  5/30/2019                                         #
#                                                                             #
#  Description: The driver to create NetCDF time slice files from Army Crops  #
#               of Engineers real-time observations                           #
#                                                                             #
#  12/11/2024   OWP   Update code to use JSON file instead of XML             #
#                                                                             #
###############################################################################

import os, sys, time, urllib, getopt
import logging
from string import *
import xml.etree.ElementTree as etree
from datetime import datetime, timedelta
from TimeSlice import TimeSlice
from Observation import Observation, All_Observations
from CWMS_Sites import CWMS_Sites
from ACE_Observation import ACE_Observation
from EmptyDirOrFileException import EmptyDirOrFileException
#import Tracer

"""
   The driver to parse downloaded ACE JSON observations and 
   create time slices and write to NetCDF files
   Author: Zhengtao Cui (Zhengtao.Cui@noaa.gov)
   Date: May 30, 2019
"""
def main(argv):
   """
     function to get input arguments
   """
   inputdir = ''
   try:
           opts, args = getopt.getopt(argv,"hi:o:s:",["idir=", "odir=", "sites="])
   except getopt.GetoptError:
      print( 'make_time_slice_from_ace.py -i <inputdir> -o <outputdir> -s <sitefile>' )
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print( \
           'make_time_slice_from_ace.py -i <inputdir> -o <outputdir> -s <sitefile>' )
         sys.exit()
      elif opt in ('-i', "--idir"):
         inputdir = arg
         if not os.path.exists( inputdir ):
                 raise RuntimeError( 'FATAL ERROR: inputdir ' + \
                                 inputdir + ' does not exist!' )
      elif opt in ('-o', "--odir" ):
         outputdir = arg
         if not os.path.exists( outputdir ):
                 raise RuntimeError( 'FATAL ERROR: outputdir ' + \
                                 outputdir + ' does not exist!' )
      elif opt in ('-s', "--sites" ):
         sitefile = arg
         if not os.path.exists( sitefile ):
                 raise RuntimeError( 'FATAL ERROR: sitefile ' + \
                                 sitefile + ' does not exist!' )
  
   return (inputdir, outputdir, sitefile)

t0 = time.time()

logging.basicConfig(format=\
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
                level=logging.INFO)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(\
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#logger.setFormatter(formatter)
logger.info( "System Path: " + str( sys.path ) )

if __name__ == "__main__":
   try:
      odir = main(sys.argv[1:])
   except Exception as e:
      logger.error("Failed to get program options.", exc_info=True)

indir = odir[0]
outdir = odir[1]
sitefile = odir[2]
logger.info( 'Input dir is "' + indir + '"')
logger.info( 'Output dir is "' + outdir + '"')
logger.info( 'Site file is "' + sitefile + '"')

#
# Load ACE observed JSON discharge data
#

try:
   sites=CWMS_Sites( sitefile )

   obvs = []

   if not os.path.isdir( indir ):
       raise RuntimeError( "FATAL ERROR: " + indir + \
                                   " is not a directory or does not exist. ")

   for file in os.listdir( indir ): 
       # Process json file if file size is greater than 2 bytes (not empty)
       if file.endswith( ".json" ) and os.path.getsize(indir+"/"+file) > 2:
             logger.info( 'Reading ' + indir + '/' + file + ' ... ' )
             try:
                     obvs.append( ACE_Observation( \
                                           indir + '/' + file, sites ) )
             except Exception as e:
                           logger.warning( repr( e ), exc_info=True )
                           continue
       else:
           logger.warning('Skip ' + indir + '/' + file + ' because file is empty (2 bytes filesize).\n')

   if not obvs:
       raise EmptyDirOrFileException( "Input directory " + indir + \
             " has no USACE json files or the json files are empty!")

   allobvs = All_Observations( obvs )

except EmptyDirOrFileException as e:
   logger.warning( str(e), exc_info=True)
   sys.exit(0)

except Exception as e:
   logger.error("Failed to load USACE JSON files: " + str(e), exc_info=True)
   sys.exit(3)

   
logger.info( 'Earliest time in USACE JSON: ' + \
                allobvs.timePeriodForAll()[0].isoformat() )
logger.info( 'Latest time in USACE JSON: ' +  \
                allobvs.timePeriodForAll()[1].isoformat() )

#
# Create time slices from loaded observations
#
# Set time resolution to 15 minutes
# and
# Write time slices to NetCDF files
#
try:
   timeslices = allobvs.makeAllTimeSlices( timedelta( minutes = 15 ), \
                   outdir, 'usaceTimeSlice.ncdf' )
except Exception as e:
   logger.error("Failed to make time slices: " + str(e), exc_info=True)
   logger.error("Input dir = " + indir, exc_info=True)
   sys.exit(4)

logger.info( "Total number of timeslices: " + str( timeslices ) )
logger.info( "Program finished in: " + \
                "{0:.1f}".format( (time.time() - t0) / 60.0 ) + \
                 " minutes" )
