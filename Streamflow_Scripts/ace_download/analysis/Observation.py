#!/usr/bin/env python
###############################################################################
#  Module name: Observation
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:  5/24/2019                                                    #
#                                                                             #
#  Last modification date:  
#                                                                             #
#  Description: Abstract the observed real-time stream flow
#                                                                             #
###############################################################################

import os, logging
from string import *
from datetime import datetime, timedelta
import dateutil.parser
import pytz
#import iso8601
#import Tracer
from abc import ABCMeta, abstractmethod, abstractproperty
from TimeSlice import TimeSlice


class Observation:
        """
           Abstract real-time flow time series.
        """        
        __metaclass__ = ABCMeta

#        @abstractproperty
#        def source(self):
#            pass
#
#        @abstractproperty
#        def stationID(self):
#            pass
#
#        @abstractproperty
#        def stationName(self):
#            pass
#
#        @abstractproperty
#        def obvPeriod(self):
#            pass
#
#        @abstractproperty
#        def unit(self):
#            pass
#
#        @abstractproperty
#        def timeValueQuality(self):
#            pass

        @property
        def source(self):
            return self._source

        @source.setter
        def source(self, s):
            self._source = s

        @property
        def stationID(self):
            return self._stationID

        @stationID.setter
        def stationID(self, s):
            self._stationID = s
         
        @property
        def stationName(self):
            return self._stationName

        @stationName.setter
        def stationName(self, s):
            self._stationName = s

        @property
        def obvPeriod(self):
            return self._obvPeriod

        @obvPeriod.setter
        def obvPeriod(self, p):
            self._obvPeriod = p

        @property
        def unit(self):
            return self._unit

        @unit.setter
        def unit(self, u):
            self._unit = u

        @property
        def timeValueQuality(self):
            return self._timeValueQuality

        @timeValueQuality.setter
        def timeValueQuality(self, tvq):
            self._timeValueQuality = tvq 

        @abstractmethod
        def __init__(self, filename ):
            pass
        

        def getTimeValueAt(self, at_time, resolution = timedelta() ):
                """
                   Get the closest time-value pair for a given time 
               
                   Input: at_time - the given time
                      resolution - the tolerance time period around the
                                   given time

                   Return:  Tuple of a time-value pair
                """

                closestTimes = []
                distances = []
                if at_time in self.timeValueQuality:
                        return ( at_time, self.timeValueQuality.get( at_time ) )
#                for k in sorted( self.timeValueQuality ):
                for k in self.timeValueQuality:
                        if ( abs( k - at_time ) <=  resolution / 2 ):
                           closestTimes.append( k )
                           distances.append( abs( k - at_time ) )
                if not closestTimes:
                        return None
                else:
                        closest = [ x for y, x in \
                                sorted( zip( distances, closestTimes ) ) ][ 0 ]

                        return ( closest, self.timeValueQuality.get( closest ) )
                    

class All_Observations:
        "Store all obvserved data"
        def __init__(self, obvs ):
           """
              Initialize the All_Objections object for a given
             Observation 

              Input: list of Observation object
           """
           self.logger = logging.getLogger(__name__)
           self.observations = obvs

           if not self.observations:
              raise RuntimeError( "FATAL ERROR: has no data")

           self.index = -1

           self.timePeriod = self.observations[0].obvPeriod
           for obv in self.observations:
               if self.timePeriod[0] > obv.obvPeriod[ 0 ]:
                   self.timePeriod = ( obv.obvPeriod[ 0 ], \
                                               self.timePeriod[ 1 ])

               if self.timePeriod[1] < obv.obvPeriod[ 1 ]:
                   self.timePeriod = ( self.timePeriod[ 0 ], \
                                               obv.obvPeriod[ 1 ] )

        def __iter__(self ):
            """
               The iterator
            """
            return self

        def __next__( self ):
           """
               The next Observation object
           """
           if self.index == len( self.observations ) - 1:
              self.index = -1
              raise StopIteration
           self.index = self.index + 1
           return self.observations[ self.index ]

        def timePeriodForAll( self ):
                """
                  Get the earlist and latest time for all observations

                  Return: Tuple of start time and end time
                """
                return self.timePeriod

        def makeTimeSlice( self, timestamp, timeresolution ):
            """
               Create one time slice for a given time and resolution

               Input: timestamp - the given time
                      timeresolution - the resolution

               Return: A time slice object
            """
            station_time_value_list = []
            for obv in self:
               closestObv = obv.getTimeValueAt( timestamp, timeresolution ) 
               if closestObv:
                 station_time_value_list.append( \
                     ( obv.stationID, closestObv[ 0 ], \
                        # value              quality
                       closestObv[ 1 ][ 0 ], closestObv[ 1 ][ 1 ] ) ) 
           
#            Tracer.theTracer.run( 'timeSlice = TimeSlice( timestamp, timeresolution, station_time_value_list )' )
            timeSlice = TimeSlice( timestamp, \
                            timeresolution, \
                            station_time_value_list )
            return timeSlice

        def makeAllTimeSlices( self, timeresolution, outdir, suffix='usaceTimeSlice.ncdf' ):
            """
               Create the time slice NetCDF files for all USACE observations

               Input: timeresolution - resolution
                      outdir - the output directory

               Return: Total number of time slice files created
            """

            # the time resultions must divide 60 minutes with on remainder                  
            if 3600 % timeresolution.seconds != 0:
               raise RuntimeError( "FATAL ERROR: Time slice resolution must "
                               "divide 60 minutes with no remainder." )

            startTime = datetime( self.timePeriod[ 0 ].year,
                                  self.timePeriod[ 0 ].month,
                                  self.timePeriod[ 0 ].day,
                                  self.timePeriod[ 0 ].hour )

            while startTime < self.timePeriod[ 0 ]:
                    startTime += timeresolution

            if startTime > self.timePeriod[ 1 ]: 
               raise RuntimeError(                  \
                            "FATAL ERROR: observation time period wrong! " )

            count = 0
            while startTime <= self.timePeriod[ 1 ]:
                    self.logger.info( "making time slice for " + \
                                    startTime.isoformat() )
#                    Tracer.theTracer.run(    \
#                            'oneSlice = makeTimeSlice( startTime, timeresolution )' )

                    oneSlice = self.makeTimeSlice( startTime, timeresolution )
                    self.logger.info( "Time slice:  " + \
                             outdir + '/' +oneSlice.getSliceNCFileName( suffix ) )
                    if ( not oneSlice.isEmpty() ):
                        updatedOrNew = True
                        slicefilename = outdir + '/' +oneSlice.getSliceNCFileName( suffix ) 
                        if os.path.isfile( slicefilename ):
#                           Tracer.theTracer.run( \
#                     'oldslice = TimeSlice.fromNetCDF( slicefilename )' )
                           oldslice = TimeSlice.fromNetCDF( slicefilename )
                           updatedOrNew = oneSlice.mergeOld( oldslice )

                        if updatedOrNew:
                          oneSlice.toNetCDF( outdir, suffix )
                          self.logger.info( oneSlice.getSliceNCFileName( suffix ) + \
                                                             " updated!" )
                        else:
                          self.logger.info( oneSlice.getSliceNCFileName( suffix ) + \
                                                            " not updated!" )
#                        oneSlice.print_station_time_value()
                        count = count + 1

                    startTime += timeresolution

#            for eachSlice in allTimeSlices:
#                    print "-----------------------------"
#                    eachSlice.print_station_time_value()
#                    print eachSlice.getSliceNCFileName()
#
#            return allTimeSlices
            return count
