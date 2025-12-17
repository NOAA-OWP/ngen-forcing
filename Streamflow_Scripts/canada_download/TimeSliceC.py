import os, sys, time, math
from string import *
from datetime import datetime, timedelta
import calendar
import netCDF4 
import numpy as np
import pytz

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    '''Python 2 implementation of Python 3.5 math.isclose()
       https://hg.python.org/cpython/file/tip/Modules/mathmodule.c#l1993'''
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

def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = dict()
    for o in intersect_keys:
        f0 = d1[o][0] == d2[o][0]
        f1 = isclose( d1[o][1], d2[o][1],abs_tol=0.01 )
        f2 = isclose( d1[o][2], d2[o][2])
        if not f0 or not f1 or not f2:
            modified[o] = (d1[o], d2[o])
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same

class TimeSliceC:
    """ Description: Store one time slice data (from Canadian data)
           Author: Tim Hunter (tim.hunter@noaa.gov) drawing HEAVILY on
           original code by Zhengtao Cui (Zhengtao.Cui@noaa.gov)
           Date: Feb 15, 2018
    """
    stationIdStrLen = 15
    stationIdLong_name = "WSC station id padded to length 15"
    timeStrLen = 19
    timeUnit = "UTC"
    timeLong_name = "YYYY-MM-DD_HH:mm:ss UTC"
    dischargeUnit = "m^3/s"
    dischargeLong_name = "Discharge.cubic_meters_per_second"
    dischargeQualityUnit = "-"
    dischargeQualityLong_name = \
            "Discharge quality 0 to 100 to be scaled by 100."

    def __init__(self, time_stamp, resolution, station_time_value ):
        # insure that time stamp is tz-aware.  Time is already UTC.
        self.centralTimeStamp = time_stamp.replace(tzinfo=pytz.UTC)
        self.sliceTimeResolution = resolution
        self.obvStationTimeValue = station_time_value

    def isEmpty( self ):
        return not self.obvStationTimeValue

    def print_station_time_value( self ):
        for e in self.obvStationTimeValue:
                    print( "Slice: central time: ", \
                self.centralTimeStamp.isoformat(), \
                e[0], e[1].isoformat(), e[2] )

    def getStationIDs( self ):
        stationL = []
        for e in self.obvStationTimeValue:
            stationL.append( list( e[ 0 ][4:] ) )
            for s in stationL:
                if len(s) < self.stationIdStrLen:
                    for i in range( len(s), self.stationIdStrLen ):
                        s.insert(0, ' ' )
                elif len(s) > self.stationIdStrLen:
                    s = s[0:self.stationIdStrLen - 1] 
        return stationL

    def getDischargeValues( self ):
        values = []
        for e in self.obvStationTimeValue:
            values.append( e[ 2 ] )
        return values

    def getDischargeTimes( self ):
        obvtimes = []
        for e in self.obvStationTimeValue:
            obvtimes.append(list(e[1].strftime("%Y-%m-%d_%H:%M:00")))
        return obvtimes

    def getQueryTimes( self ):
        qtimes = []
        for e in self.obvStationTimeValue:
            qtimes.append( calendar.timegm( e[ 1 ].utctimetuple() ) )
        return qtimes

    def getSliceNCFileName( self ):
        tval = (self.sliceTimeResolution.days * 24 * 60) + (self.sliceTimeResolution.seconds // 60)
        filename = (self.centralTimeStamp.strftime("%Y-%m-%d_%H_%M_00.") +
               str(int(tval)).zfill(2) + "min.wscTimeSlice.ncdf")
        return filename 

    def getDischargeQuality( self ):
        dq = []
        for e in self.obvStationTimeValue:
            dq.append( e[ 3 ] )
        return dq
        
    def toNetCDF( self, outputdir = './' ):
        fname = outputdir + '/' + self.getSliceNCFileName()
        nc_fid = netCDF4.Dataset(fname, 'w', format='NETCDF4' )
        nc_fid.createDimension( 'stationIdStrLen', self.stationIdStrLen ) 
        nc_fid.createDimension( 'stationIdInd', None ) 
        nc_fid.createDimension( 'timeStrLen', self.timeStrLen ) 
        stationId = nc_fid.createVariable( 'stationId', 'S1',\
                             ('stationIdInd', 'stationIdStrLen') )
        stationId.setncatts( {'long_name' : self.stationIdLong_name, \
                              'units' : '-'} )

        time = nc_fid.createVariable( 'time', 'S1',\
                              ('stationIdInd', 'timeStrLen' ) )
        time.setncatts( {'long_name' : self.timeLong_name, \
                         'units' : self.timeUnit} )

        discharge = nc_fid.createVariable( 'discharge', 'f4',\
                              ('stationIdInd', ) )
        discharge.setncatts( {'long_name' : self.dischargeLong_name, \
                              'units' : self.dischargeUnit} )
        discharge_quality = \
                nc_fid.createVariable( 'discharge_quality', 'i2',\
                ('stationIdInd', ) )
        discharge_quality.setncatts( {'long_name' : \
                            self.dischargeQualityLong_name, \
                            'units' : self.dischargeQualityUnit,  \
                            'multfactor' : '0.01' } )
        queryTime = nc_fid.createVariable( 'queryTime', 'i4',\
                              ('stationIdInd', ) )

        queryTime.setncatts( { 'units' :  \
               'seconds since 1970-01-01 00:00:00 local TZ' } )
     
        tres = self.sliceTimeResolution.days * 24 * 60 + \
               self.sliceTimeResolution.seconds // 60
        nc_fid.setncatts( { 'fileUpdateTimeUTC':  \
                datetime.utcnow().strftime( "%Y-%m-%d_%H:%M:00" ), \
                'sliceCenterTimeUTC' : \
                self.centralTimeStamp.strftime( "%Y-%m-%d_%H:%M:00" ),\
                'sliceTimeResolutionMinutes' :  \
                str(int(tres)).zfill(2) } ) 
                          
        discharge[ : ] = self.getDischargeValues()
        queryTime[ : ] = self.getQueryTimes()

        stations = self.getStationIDs()
        stationId[ : ] = stations

        time[ : ] = self.getDischargeTimes()
        discharge_quality[:] = self.getDischargeQuality()

        nc_fid.close()

    @classmethod
    def fromNetCDF( self, ncfilename ):
        nc_fid = netCDF4.Dataset( ncfilename, 'r' )
        timestamp = datetime.strptime( \
                nc_fid.getncattr( 'sliceCenterTimeUTC' ), \
                                  "%Y-%m-%d_%H:%M:00" )

        time_resol = timedelta( minutes = \
                 int( nc_fid.getncattr( 'sliceTimeResolutionMinutes' ) ) )

        stations = netCDF4.chartostring( \
                     nc_fid.variables[ 'stationId'][ : ] )

        discharge = nc_fid.variables[ 'discharge'][ : ]

        queryTime = nc_fid.variables[ 'queryTime'][ : ]

        quality = nc_fid.variables[ 'discharge_quality'][ : ]

        stationTimeValue = []
        for s, d, q, qual in zip( stations, discharge, queryTime, quality ):
            stationTimeValue.append( \
                ( 'CAN.' + s.strip(), \
                datetime.utcfromtimestamp( q ).replace(tzinfo=pytz.UTC), \
                d, qual ) )

        nc_fid.close()
        return self( timestamp, time_resol, stationTimeValue )

    def mergeOld( self, oldTimeSlice ):
        if self.centralTimeStamp != oldTimeSlice.centralTimeStamp or \
            self.sliceTimeResolution != oldTimeSlice.sliceTimeResolution:
            print( 'new_cts=', self.centralTimeStamp )
            print( 'old_cts=', oldTimeSlice.centralTimeStamp )
            print( 'new_res=', self.sliceTimeResolution )
            print( 'old_res=', oldTimeSlice.sliceTimeResolution )
            raise RuntimeError( "FATAL ERROR: the two time slices " + 
                  " differ, not merging ..." )
        else:
            site_time_value = dict()
        for e in self.obvStationTimeValue:
            site_time_value[ e[ 0 ] ] = ( e[1], e[2], e[3] )
            old_site_time_value = dict()

        for e in oldTimeSlice.obvStationTimeValue:
            old_site_time_value[ e[ 0 ] ] = ( e[1], e[2], e[3] )
            
        added, removed, modified, same = (
            dict_compare(site_time_value, old_site_time_value ))

        if not added and not modified :
            return False

        old_site_time_value.update( site_time_value ) 
        self.obvStationTimeValue = []
        for site in old_site_time_value:
            self.obvStationTimeValue.append( ( site, \
                    old_site_time_value[ site ][ 0 ], \
                    old_site_time_value[ site ][ 1 ], \
                    old_site_time_value[ site ][ 2 ] ) )
        return True
