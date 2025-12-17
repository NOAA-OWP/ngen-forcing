###############################################################################
#  Module name: SCHISMGrid                                                    #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:  12/24/2024                                          #
#                                                                             #
#  Last modification date:                                                    # 
#                                                                             #
#  Description: Manages a SCHISM output file                                  #
#                                                                             #
###############################################################################

import glob, numpy as np, netCDF4 as nc
#from cftime import date2num, num2date, timedelta
from datetime import datetime, timedelta

class SCHISMOutput:
        """
           SCHISM output file
        """        

        def __init__(self, outputfile ): 
           self._source = outputfile

           with nc.Dataset(outputfile,'r') as data:

              self._depth = data.variables['depth'][:]
              self._elev = data.variables['elevation'][0][:]

              self._lons = data.variables['SCHISM_hgrid_node_x'][:]
              self._lats = data.variables['SCHISM_hgrid_node_y'][:]

              self._nnodes = data.dimensions[ 'nSCHISM_hgrid_node' ].size

              # find base times
              y, m, d, h, _ = [int(float(e)) for e in data.variables['time'].base_date.split()]
              basetime = datetime(y, m, d, h)
              self._valid_time = basetime + timedelta( seconds = int(data.variables['time'][0].astype('i4')) ) 

        @property
        def source(self):
            return self._source

        @source.setter
        def source(self, s):
            self._source = s

        @property
        def lats(self):
            return self._lats

        @lats.setter
        def lats(self, l):
            self._lats = l

        @property
        def lons(self):
            return self._lons

        @lons.setter
        def lons(self, l):
            self._lons = l

        @property
        def nnodes(self):
            return self._nnodes

        @nnodes.setter
        def nnodes(self, n):
            self._nnodes = n

        @property
        def elev(self):
            return self._elev

        @elev.setter
        def elev(self, z):
            self._elev = z

        @property
        def depth(self):
            return self._depth

        @depth.setter
        def depth(self, d):
            self._depth = d

        @property
        def valid_time(self):
            return self._valid_time

        @valid_time.setter
        def valid_time(self, t):
            self._valid_time = t
