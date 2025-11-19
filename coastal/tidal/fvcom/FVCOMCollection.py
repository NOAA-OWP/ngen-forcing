###############################################################################
#  Module name: FVCOMCollection                                               #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:  12/23/2024                                          #
#                                                                             #
#  Last modification date:                                                    # 
#                                                                             #
#  Description: A collection of The Great Lakes FVCOM product from NOAA's     #
#    Center for Operational Oceanographic Products and Services (CO-OPS).     #
#                                                                             #
###############################################################################

import os, logging
import numpy as np
from string import *
from datetime import datetime, timedelta
import netCDF4 as nc
from FVCOM import FVCOM

class FVCOMCollection:
        """
           A Collection of the Great Lakes FVCOM products
        """        

        TIMESTEP_IN_SECS = 3600

        @staticmethod
        def makeFVCOMFilename( domain, pdy, cyc, type, casthour, casttype ):

            if int( pdy ) < int( FVCOM.FIRST_STATIONFILE_DATE ):
                if type != "fields":
                    return None
                else:
                    return f'nos.{domain}.fields.n{casthour:03}.{pdy}.t{cyc:02}z.nc'
            elif int( pdy ) < int( FVCOM.NEW_NAMING_DATE ):
                if type == "fields":
                    return f'nos.{domain}.fields.n{casthour:03}.{pdy}.t{cyc:02}z.nc'
                elif type == "stations":
                    return f'nos.{domain}.stations.nowcast.{pdy}.t{cyc:02}z.nc'
                else:
                    return None
            else:
                if type == "fields" or type == "regulargrid":
                    return f'{domain}.t{cyc:02}z.{pdy}.{type}.{ "f" if casttype == "forecast" else "n"}{casthour:03}.nc'
                elif type == "stations":
                    return f'{domain}.t{cyc:02}z.{pdy}.stations.{casttype}.nc'

        def __init__(self, start_t, end_t, fvcomdir, domain, type ):
            self.starttime = start_t
            self.endtime = end_t
            self.domain = domain
            self.type = type
            self.dir = fvcomdir
            self.filelist = self.getFilenameList();
            

        @property
        def starttime(self):
            return self._starttime

        @starttime.setter
        def starttime(self, s):
            self._starttime = s

        @property
        def endtime(self):
            return self._endtime

        @endtime.setter
        def endtime(self, e):
            self._endtime = e

        @property
        def dir(self):
            return self._dir

        @dir.setter
        def dir(self, d):
            self._dir = d

        @property
        def domain(self):
            return self._domain

        @domain.setter
        def domain(self, d):
            self._domain = d

        @property
        def type(self):
            return self._type

        @type.setter
        def type(self, t):
            self._type = t

        @property
        def filelist(self):
            return self._filelist

        @filelist.setter
        def filelist(self, f):
            self._filelist = f

        def getFilenameList( self ):
            delta = timedelta(seconds = FVCOMCollection.TIMESTEP_IN_SECS )
            t = self.starttime 
            filelist = []
            while ( t < self.endtime + timedelta( hours = 6 )):
                pdy = t.strftime( "%Y%m%d" )
                cyc = int( t.strftime( "%-H" ) )
                casthour = int(cyc) % 6
                if casthour != 0:
                    cyc = int( cyc / 6 ) * 6

                filelist.append( os.path.join(self.dir, \
                       FVCOMCollection.makeFVCOMFilename( self.domain, pdy, cyc, self.type, casthour + 1, 'nowcast') ))
                t += delta 

#            for f in filelist:
#                print( f )
#            exit()
            return filelist

        def getLons( self ):
            fvcom = FVCOM( self.filelist[0] )
            return np.ma.getdata((fvcom.lons+180) % 360 - 180)

        def getLats( self ):
            fvcom = FVCOM( self.filelist[0] )
            return np.ma.getdata(fvcom.lats)

        def getRefTime( self ):
            fvcom = FVCOM( self.filelist[0] )
            return fvcom.ref_time()

        def getTimes( self ):
            times = np.empty(len(self.filelist),dtype=float)
            for i in range(len(self.filelist)):
                fvcom = FVCOM( self.filelist[ i ] )
                times[i] = fvcom.time.data[0]
            return times

        def getTotalNumNodes( self ):
            return len(self.getLons())

        def getZetas( self ):
            zetas = np.empty((len(self.filelist),self.getTotalNumNodes()),dtype=float)
            for i in range(len(self.filelist)):
                fvcom = FVCOM( self.filelist[ i ] )
                zetas[i,:] = fvcom.zeta.data
            return zetas

