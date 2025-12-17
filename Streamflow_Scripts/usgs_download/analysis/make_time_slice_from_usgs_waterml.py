#!/usr/bin/env python
###############################################################################
#  File name: make_time_slice_from_usgs_waterml.py                            #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  7/12/2017                                         #
#                                                                             #
#  Description: The driver to create NetCDF time slice files from USGS        #
#               real-time observations                                        #
#                                                                             #
#  06/11/2024  ChamP  - Use USGS .json file instead of .xml file              #
###############################################################################

import os, sys, time, getopt
import logging
from string import *
import xml.etree.ElementTree as etree
from datetime import datetime, timedelta
from USGS_Observation import USGS_Observation
from TimeSlice import TimeSlice
from Observation import Observation, All_Observations
from EmptyDirOrFileException import EmptyDirOrFileException
#import Tracer

"""
   The driver to parse downloaded waterML Json observations and 
   create time slices and write to NetCDF files
   Author: Zhengtao Cui (Zhengtao.Cui@noaa.gov)
   Date: Aug. 26, 2015
"""
def main(argv):
   """
     function to get input arguments
   """
   inputdir = ''
   try:
           opts, args = getopt.getopt(argv,"hi:o:",["idir=", "odir="])
   except getopt.GetoptError:
      print('make_time_slice_from_usgs_waterml.py -i <inputdir> -o <outputdir>') 
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print(   \
           'make_time_slice_from_usgs_waterml.py -i <inputdir> -o <outputdir>') 
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
  
   return (inputdir, outputdir)

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
logger.info( 'Input dir is "' + indir + '"')
logger.info( 'Output dir is "' + outdir + '"')

#
# Load USGS observed JSON discharge data
#

try:
   usgsobvs = []

   if not os.path.isdir( indir ):
           raise RuntimeError( "FATAL ERROR: " + indir + \
                                   " is not a directory or does not exist. ")
   for file in os.listdir( indir ): 
        if file.endswith( ".json" ) or file.endswith( ".xml" ) or file.endswith( ".csv" ):
             logger.info( 'Reading ' + indir + '/' + file + ' ... ' )
             try:
                     usgsobvs.append( USGS_Observation( \
                                           indir + '/' + file ) )
             except Exception as e:
                           logger.warning( repr( e ), exc_info=True )
                           continue

   if not usgsobvs:
         raise EmptyDirOrFileException( "Input directory " + indir + \
             " has no USGS json files, or the files are empty, or no discharge values!")

   allobvs = All_Observations( usgsobvs )

except EmptyDirOrFileException as e:
   logger.warning( str(e), exc_info=True)
   sys.exit(0)

except Exception as e:
   logger.error("Failed to load WaterJson files: " + str(e), exc_info=True)
   sys.exit(3)

logger.info( 'Earliest time in WaterJson: ' + \
                allobvs.timePeriodForAll()[0].isoformat() )
logger.info( 'Latest time in WaterJson:: ' +  \
                allobvs.timePeriodForAll()[1].isoformat() )

#
# Create time slices from loaded observations
#
# Set time resolution to 15 minutes
# and
# Write time slices to NetCDF files
#
try:
   timeslices = allobvs.makeAllTimeSlices( timedelta( minutes = 15 ), outdir )
except Exception as e:
   logger.error("Failed to make time slices: " + str(e), exc_info=True)
   logger.error("Input dir = " + indir, exc_info=True)
   sys.exit(3)

logger.info( "Total number of timeslices: " + str( timeslices ) )
logger.info( "Program finished in: " + \
                "{0:.1f}".format( (time.time() - t0) / 60.0 ) + \
                 " minutes" )
