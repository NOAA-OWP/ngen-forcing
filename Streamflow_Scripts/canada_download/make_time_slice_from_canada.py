import os, sys, time, urllib, getopt
import logging
from string import *
import xml.etree.ElementTree as etree
from datetime import datetime, timedelta
from WSC_Observation import WSC_Observation, All_WSC_Observations
from TimeSliceC import TimeSliceC
from EmptyDirOrFileException import EmptyDirOrFileException

"""
   The driver to parse downloaded hydrologic data files from
   Environment and Climate Change Canada (Water Survey of Canada), 
   and then to create time slices and write to NetCDF files.
   Author: Tim Hunter (tim.hunter@noaa.gov) drawing heavily on 
   the prior work done by Zhengtao Cui for the USGS data.
   Date: February 2018
"""
def main(argv):
   """
     function to get input arguments
   """
   inputdir = ''
   try:
           opts, args = getopt.getopt(argv,"hi:o:",["idir=", "odir="])
   except getopt.GetoptError:
      print( 'make_time_slice_from_canada.py -i <inputdir> -o <outputdir>'  )
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print(   \
           'make_time_slice_from_canada.py -i <inputdir> -o <outputdir>' )
         sys.exit()
      elif opt in ('-i', "--idir"):
         inputdir = arg
         if not os.path.exists( inputdir ):
                 raise RuntimeError( 'FATAL Error: inputdir ' + \
                                 inputdir + ' does not exist!' )
      elif opt in ('-o', "--odir" ):
         outputdir = arg
         if not os.path.exists( outputdir ):
                 raise RuntimeError( 'FATAL Error: outputdir ' + \
                                 outputdir + ' does not exist!' )
  
   return (inputdir, outputdir)

logging.basicConfig(format=\
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
                level=logging.INFO)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(\
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#logger.setFormatter(formatter)
logger.info( "System Path: " + str( sys.path ) )

if __name__ == "__main__":
   odir = main(sys.argv[1:])

indir = odir[0]
outdir = odir[1]

#
#  Load discharge data as downloaded from ECCC datamart.  Current URL
#  for a parent directory is http://dd.weather.gc.ca/hydrometric/csv/
#  Files are found further down that tree.
#
try:
   allobvs = All_WSC_Observations( indir )
except EmptyDirOrFileException as e:
   logger.warning( str(e), exc_info=True)
   sys.exit(0)
except Exception as e:
   logger.error("Failed to load Canadian CSV files: " + str(e), exc_info=True)
   sys.exit(3)

print( 'earliest time: ', allobvs.timePeriodForAll()[0] )
print( 'latest time: ', allobvs.timePeriodForAll()[1] )

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
   logger.error("Failed to make time slices: " + str(e) , exc_info=True)
   logger.error("Input dir = " + indir, exc_info=True)
   sys.exit(3)

print( "total number of timeslices: ", timeslices )
