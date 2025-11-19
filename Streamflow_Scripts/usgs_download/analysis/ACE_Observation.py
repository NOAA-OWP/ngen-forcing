###############################################################################
#  Module name: ACE_Observation
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  05/28/2019                                         #
#                                                                             #
#  Description: manage data in a ACE CWMS xml file                       #
#                                                                             #
###############################################################################

import os, sys, time, csv, re
import logging
from string import *
from collections import OrderedDict
from datetime import datetime, timedelta
import dateutil.parser
import pytz
#import iso8601
import xml.etree.ElementTree as etree
from TimeSlice import TimeSlice
#import Tracer
from Observation import Observation
from CWMS_Sites import CWMS_Sites

def parseDuration( period ):
        regex  = re.compile('(?P<sign>-?)P(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?')

        # Fetch the match groups with default value of 0 (not None)
        duration = regex.match(period).groupdict(0)

        # Create the timedelta object from extracted groups
        delta = timedelta(days=int(duration['days']) + \
                        (int(duration['months']) * 30) + \
                        (int(duration['years']) * 365), \
                        hours=int(duration['hours']), \
                        minutes=int(duration['minutes']), \
                        seconds=int(duration['seconds']))

        if duration['sign'] == "-":
                    delta *= -1

        return delta

class ACE_Observation(Observation):
        """
           Store one USGS WaterML2.0 data
        """        
        def __init__(self, cwmsxmlfilename, cwmssites ):
           """
              Initialize the ACE_Observation object with a given
              filename
           """
           self.source = cwmsxmlfilename
           self.timeValueQuality = OrderedDict()
           self._sites = cwmssites
           if cwmsxmlfilename.endswith( '.xml' ):
              self.loadCWMSxml( cwmsxmlfilename )
           else:
              raise RuntimeError( "FATAL ERROR: Unknow file type: " + \
                                 cwmsxmfilename )

        def loadCWMSxml(self, xmlfilename ):
           """
              Read real-time stream flow data from a given CWMS XML file

              Input: xmlfilename - the CWMS xml filename
           """        
           try:
                obvwml = etree.parse( xmlfilename )
                root= obvwml.getroot()
                name_1 = root.find('query-info').find('requested-item')\
                             .find('name').text
                timeseries = root.find('time-series')
                office = timeseries.find('office').text 
                self.stationName = office + "." + name_1
                self.stationID = self._sites.getIndex(office, name_1 )

                regularIntervalValues = timeseries.find('regular-interval-values')
                if regularIntervalValues is not None:
                   self.parseRegularIntervalValues( regularIntervalValues )
                else: 
                   irregularIntervalValues = timeseries.find('irregular-interval-values')
                   self.parseIrregularIntervalValues( irregularIntervalValues)

           except Exception as e:
               raise RuntimeError( "WARNING: parsing XML error: " + str( e )\
                               + ": " + xmlfilename + " skipping ..." )

           self.stationName = office + '.' + name_1
           self.obvPeriod =  list( self.timeValueQuality.keys() )[0], \
                             list( self.timeValueQuality.keys() )[-1]

           unitConvertToM3perSec = self.getUnitConvertToM3perSec()

           self.timeValueQuality = dict(map( \
                   lambda kv: (kv[0], (kv[1][0] * unitConvertToM3perSec, \
                                       kv[1][1])),\
                                  iter( self.timeValueQuality.items()) ))
           self.unit = 'm3/s'

#           for k, v in self.timeValueQuality.items():
#                   print(k, v)

        def parseRegularIntervalValues(self, regularInterval):
              self.unit = regularInterval.get('unit')

              interval = parseDuration( regularInterval.get('interval') )

              for seg in regularInterval.findall('segment'):
                      beginTime = \
                         dateutil.parser.parse( seg.get('first-time')) \
                          .astimezone(pytz.utc).replace(tzinfo=None)
                      for s in seg.text.strip().split('\n'):
                              self.timeValueQuality[ beginTime ] = \
                                              ( float(s.split(' ')[0]), \
                         self.calculateDataQuality( float(s.split(' ')[1] ) ) )

                              beginTime += interval 


        def parseIrregularIntervalValues(self, irregularInterval):
              self.unit = irregularInterval.get('unit')
              for s in irregularInterval.text.strip().split('\n'):
                      words = s.split(' ')
                      t = dateutil.parser.parse( words[0] ) \
                          .astimezone(pytz.utc).replace(tzinfo=None)
                      self.timeValueQuality[ t ] = \
                                              ( float(words[1]), \
                              self.calculateDataQuality(float(words[2] ) ) )
                    #  print( t, self.timeValueQuality[ t ] )

 
        def getUnitConvertToM3perSec(self):
                if self.unit == 'cfs':
                  unitConvertToM3perSec = 0.028317
                elif self.unit == 'CMS':
                  unitConvertToM3perSec = 1.0
                else:
                  raise RuntimeError( "FATAL ERROR: Unit " + self.unit + \
                                   " is not known. ")
                return unitConvertToM3perSec
        def calculateDataQuality(self, value ):
                return 100.0
