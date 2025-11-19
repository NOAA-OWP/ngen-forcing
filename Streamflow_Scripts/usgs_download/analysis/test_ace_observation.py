###############################################################################
#  File name: make_time_slice_from_usgs_waterml.py
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
###############################################################################

import os, sys, time, urllib, getopt
import logging
from string import *
import xml.etree.ElementTree as etree
from datetime import datetime, timedelta
from ACE_Observation import ACE_Observation
from TimeSlice import TimeSlice
from Observation import Observation, All_Observations
from CWMS_Sites import CWMS_Sites
#import Tracer

"""
   The driver to parse downloaded waterML 2.0 observations and 
   create time slices and write to NetCDF files
   Author: Zhengtao Cui (Zhengtao.Cui@noaa.gov)
   Date: Aug. 26, 2015
"""
def main(argv):
    """
      function to get input arguments
    """
    sites=CWMS_Sites( "./CWMS_outflow_sites_263_index.csv")
    aceflow=ACE_Observation( "./test_data/SWT_AMES.Flow.Inst.1Hour.0.Ccp-Rev.xml", sites)

    irregularaceflow=ACE_Observation( "./test_data/SWF_TX08008-Gated_Total.Flow-Out.Ave.~1Day.1Day.Rev-SWF-REGI.xml", sites)

if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print("Failed to get program options." + str(e))


