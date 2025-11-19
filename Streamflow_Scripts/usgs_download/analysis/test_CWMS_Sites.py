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
from datetime import datetime, timedelta
from CWMS_Sites import CWMS_Sites
import csv
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
    index = sites.getIndex( "SAJ", "S135-Pump.Flow.Inst.1Hour.0.SFWMD-WM")
    print("index = " + index)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print("Failed to get program options." + str(e))


