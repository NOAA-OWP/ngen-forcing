#! /usr/bin/env python

import os, sys, time, urllib, getopt
import logging
import argparse
import math
from datetime import datetime
import dateutil.parser
import numpy as np
import netCDF4
import FVCOM
from FVCOMCollection import FVCOMCollection
from SCHISMGrid import SCHISMGrid
from SCHISMOceanMaker import SCHISMOceanMaker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fvcom_dir', type=str, help='FVCOM inputfile directory')
    parser.add_argument('fvcom_domain', type=str, help='FVCOM domain name, one of leofs, lmhofs, loofs, and lsofs')
    parser.add_argument('schism_grid', type=str, help='schism horizontal grid .nc file containing schism coordinates')
    parser.add_argument('start_time', type=str, help='Start time of the SCHISM simulation period in a string that is supported by the Python dateutil.parser package, for example, "2024-01-02 00:00"')
    parser.add_argument('end_time', type=str, help='End time of the SCHISM simulation period in a string that is supported by the Python dateutil.parser package, for example, "2024-01-02 23:00"')
    parser.add_argument('output', type=str, help='Output schism tidal boundary condition .nc file')

    args = parser.parse_args()

    start_time = dateutil.parser.parse( args.start_time )
    end_time = dateutil.parser.parse( args.end_time )

    collection = FVCOMCollection( start_time, end_time, 
                              args.fvcom_dir, args.fvcom_domain, "fields" )

    schm_grid = SCHISMGrid( args.schism_grid, SCHISMGrid.ZETA0[ args.fvcom_domain ] )

    ocean_maker = SCHISMOceanMaker( schm_grid )

    ocean_maker.makeOcean( collection, args.output )

if __name__ == "__main__":
   try:
      main()
   except Exception as e:
      logger.error("Failed to get program options.", exc_info=True)

