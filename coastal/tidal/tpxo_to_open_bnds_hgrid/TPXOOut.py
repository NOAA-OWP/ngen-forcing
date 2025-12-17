###############################################################################
#  Module name: TPXOOut                                                     #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Description: manage a OTPSnc output file that contains the timeseries      #
#                                                                             #
###############################################################################
import os, sys, time, math
from string import *
from datetime import datetime, timedelta
import calendar
#import xml.utils.iso8601
#from netCDF4 import Dataset
import netCDF4 
import numpy as np
import pandas as pd

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    '''
    Python 2 implementation of Python 3.5 math.isclose()
    https://hg.python.org/cpython/file/tip/Modules/mathmodule.c#l1993
    '''
    # sanity check on the inputs
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("tolerances must be non-negative")

    # short circuit exact equality -- needed to catch two infinities of
    # the same sign. And perhaps speeds things up a bit sometimes.
    if a == b:
        return True

    # This catches the case of two infinities of opposite sign, or
    # one infinity and one finite number. Two infinities of opposite
    # sign would otherwise have an infinite relative tolerance.
    # Two infinities of the same sign are caught by the equality check
    # above.
    if math.isinf(a) or math.isinf(b):
        return False

    # now do the regular computation
    # this is essentially the "weak" test from the Boost library
    diff = math.fabs(b - a)
    result = (((diff <= math.fabs(rel_tol * b)) or
             (diff <= math.fabs(rel_tol * a))) or
             (diff <= abs_tol))
    return result

#
# Compare two Python dictionary objects
#
# Input: d1 - one of the two dictionary object to be compared
#        d2 - one of the two dictionary object to be compared
#
# Return: Tuple of added element set, removed element set, modified element
#         set and the same element set,  
#
def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = dict()
    for o in intersect_keys:
            if d1[o][0] != d2[o][0] or \
                    not isclose( d1[o][1], d2[o][1],abs_tol=0.01 ) \
                    or not isclose( d1[o][2], d2[o][2]):
              modified[o] = (d1[o], d2[o] )
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same

class TPXOOut:
        """
           Description: Store one timeseries from OTPSnc predict_tide 
           Author: Zhengtao Cui (Zhengtao.Cui@rtx.com)
        """

        def __init__(self, otpsncoutfile ):
           """
              Initialize an output  object
           """

           #self.df = pd.read_csv( otpsncoutfile, skiprows=4)
           self.df = pd.read_csv( otpsncoutfile, sep='\s+', header=3, on_bad_lines='skip')

        def print( self ):
            """
               print the dataframe
            """
            print(self.df)

        def getNumberOfLocations( self ):
              """
                 Get the number of locations
                 Return: an integer
              """

              return 1

        @classmethod
        def toNetCDF( self, outputdir = './', suffix='.nc' ):
            """
              Write the time slice to a NetCDF file
              Input: outputdir - the directory where to write the NetCDF 
            """

