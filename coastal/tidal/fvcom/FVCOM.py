###############################################################################
#  Module name: FVCOM                                                         #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:  12/19/2024                                          #
#                                                                             #
#  Last modification date:                                                    # 
#                                                                             #
#  Description: The Great Lakes FVCOM product from NOAA's Center              #
#         for Operational Oceanographic Products and Services (CO-OPS).       #
#                                                                             #
###############################################################################

import os, logging
from string import *
from datetime import datetime, timedelta
import numpy as np
import re
import netCDF4
from TidesCurrentsProduct import TidesCurrentsProduct

class FVCOM(TidesCurrentsProduct):
        """
           The Great Lakes FVCOM product
        """        
        #
        #static variables
        #
        NEW_NAMING_DATE = '20240301'
        FIRST_STATIONFILE_DATE = '20230101'

        def __init__(self, fvcomfile ):
            self.name = "fvcom"
            self.source = fvcomfile
            self.parseFilename( fvcomfile )

            with netCDF4.Dataset(fvcomfile, mode='r') as ds:
                self.zeta = ds.variables['zeta'][:]
                self.time = ds.variables['time'][:]
                self.lons = ds.variables['lon'][:]
                self.lats = ds.variables['lat'][:]

        @property
        def source(self):
            return self._source

        @source.setter
        def source(self, s):
            self._source = s

        @property
        def name(self):
            return self._name

        @name.setter
        def name(self, n):
            self._name = n

        @property
        def zeta(self):
            return self._zeta

        @zeta.setter
        def zeta(self, z):
            self._zeta = z

        @property
        def pdy(self):
            return self._pdy

        @pdy.setter
        def pdy(self, p):
            self._pdy = p

        @property
        def cyc(self):
            return self._cyc

        @cyc.setter
        def cyc(self, c):
            self._cyc = c

        @property
        def domain(self):
            return self._domain

        @domain.setter
        def domain(self, d):
            self._domain = d

        #type - grid, stations, fields
        @property
        def type(self):
            return self._type

        @type.setter
        def type(self, t):
            self._type = t

        #casttime - forecast or nowcast
        @property
        def casttype(self):
            return self._casttype

        @casttype.setter
        def casttype(self, t):
            self._casttype = t

        @property
        def casthour(self):
            return self._casthour

        @casttype.setter
        def casthour(self, t):
            self._casthour = t

        @property
        def lons(self):
            return self._lons

        @lons.setter
        def lons(self, l):
            self._lons = l

        @property
        def lats(self):
            return self._lats

        @lats.setter
        def lats(self, l):
            self._lats = l

        @property
        def time(self):
            return self._time

        @time.setter
        def time(self, t):
            self._time = t

        def parseFilename(self, fvcomfile ):

            #the new filename convension
            fvcompattern = ( "(leofs|lmhofs|loofs|lsofs)\.t([0-2][0-9])z\.([0-2][0-9]{3}[01][0-9][0-3][0-9])"
                             "\.(fields|regulargrid|stations).(forecast|nowcast|f|n)([0-9]{3})?\.nc" )

            #the old filename convension
            oldfvcompattern = ( "nos\.(leofs|lmhofs|loofs|lsofs)\.(fields|stations)\.(nowcast|n)([0-9]{3})?"
                                "\.([0-2][0-9]{3}[01][0-9][0-3][0-9])\.t([0-2][0-9])z.nc" )

            results = re.match( fvcompattern,  os.path.basename( fvcomfile ) )

            if results:
               self.domain =  results.group(1)
               self.cycle =  results.group(2)
               self.pdy =  results.group(3)
               self.type =  results.group(4)

               self.casttype =  results.group(5)
               if self.casttype == 'n':
                   self.casttype = 'nowcast'
               if self.casttype == 'f':
                   self.casttype = 'forecast'

               self.casthour =  results.group(6)

            else:
                #new try the old naming convention
                results = re.match( oldfvcompattern, os.path.basename( fvcomfile ) )
                if results:
                   self.domain =  results.group(1)
                   self.type =  results.group(2)

                   self.casttype =  results.group(3)
                   if self.casttype == 'n':
                      self.casttype = 'nowcast'
                   if self.casttype == 'f':
                      self.casttype = 'forecast'

                   self.casthour =  results.group(4)
                   self.pdy =  results.group(5)
                   self.cycle =  results.group(6)

                else:
                   print( "No match!")

        def ref_time( self ):
            with netCDF4.Dataset(self.source, mode='r') as ds:
                timevar = ds.variables['time']
                ref_time = timevar.units.rstrip('UTC').rstrip()
            return ref_time
