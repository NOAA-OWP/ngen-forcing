#!/usr/bin/env python
###############################################################################
#  Module name: USGS_Observation                                              # 
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  12/20/2017                                        #
#                                                                             #
#  Description: manage data in a USGS WaterXML 2.0 file                       #
#                                                                             #
#  06/11/2024 ChamP - Added loadJSON() function to decode the USGS Json file  #
#                                                                             #
###############################################################################

import os, sys, time, csv
import logging
from string import *
from datetime import datetime, timedelta
import dateutil.parser
import pytz
#import iso8601
import xml.etree.ElementTree as etree
import json
from TimeSlice import TimeSlice
#import Tracer
from Observation import Observation


class USGS_Observation(Observation):
        """
           Store one USGS WaterML2.0 data
        """        
        def __init__(self, waterml2orcsvfilename ):
           """
              Initialize the USGS_Observation object with a given
              filename
           """
           self.source = waterml2orcsvfilename
           if waterml2orcsvfilename.endswith( '.json' ):
              self.loadJSON( waterml2orcsvfilename )
           elif waterml2orcsvfilename.endswith( '.xml' ):
              self.loadWaterML2( waterml2orcsvfilename )
           elif waterml2orcsvfilename.endswith( '.csv' ):
              self.loadCSV( waterml2orcsvfilename )
           else:
              raise RuntimeError( "FATAL ERROR: Unknow file type: " + \
                                 waterml2orcsvfilename )

        def loadJSON(self, jsonfilename ):
           """
              Read real-time stream flow data from a given Json file
              by parsing the JSON file

              Input: jsonfilename - the Json filename
           """        

           try:
               json_file = open(jsonfilename)

               # Reads .json file to Python dictionary
               json_data = json.load(json_file)
                 
               # gets data information on the USGS time series available
               timeseries = json_data['value']['timeSeries'][0]
               
               self._stationName = timeseries['sourceInfo']['siteName']
               self._stationID = timeseries['sourceInfo']['siteCode'][0]['value']
               self._unit = timeseries['variable']['unit']['unitCode']

               if self._unit == 'ft3/s':
                  unitConvertToM3perSec = 0.028317
               elif self._unit == 'm3/s':
                  unitConvertToM3perSec = 1.0
               else:
                  raise RuntimeError( "FATAL ERROR: Unit " + self._unit + \
                                      " is not known. ")
               self._unit = 'm3/s'

               # Get Discharge timeseries values
               for tsIndex in range(len(timeseries['values'])):
                  tsVals = timeseries['values'][tsIndex]['value']
                  lengthTS = len(tsVals)

                  if lengthTS > 0:
                    self._timeValueQuality = dict()

                    qualifier1 = timeseries['values'][tsIndex]['qualifier'][0]['qualifierCode']

                    beginTime = timeseries['values'][tsIndex]['value'][0]['dateTime']
                    endTime = timeseries['values'][tsIndex]['value'][lengthTS-1]['dateTime']
                    
                    self._obvPeriod = dateutil.parser.parse( beginTime )\
                                          .astimezone(pytz.utc).replace(tzinfo=None), \
                                          dateutil.parser.parse( endTime )\
                                          .astimezone(pytz.utc).replace(tzinfo=None)

                    for point in range(0, lengthTS):
                       valuetime = timeseries['values'][tsIndex]['value'][point]['dateTime']
                    
                       if valuetime and not valuetime.isspace():
                         value = timeseries['values'][tsIndex]['value'][point]['value']

                         qualifiers = timeseries['values'][tsIndex]['value'][point]['qualifiers']

                         qualifier2 = ''
                      
                         if len(qualifiers) > 0:
                            qualifier1 = timeseries['values'][tsIndex]['value'][point]['qualifiers'][0]
                            if len(qualifiers) > 1:
                               qualifier2 = timeseries['values'][tsIndex]['value'][point]['qualifiers'][1]

                         if value == '-999999':
                            self._timeValueQuality[ dateutil.parser.parse( valuetime )\
                                                .astimezone(pytz.utc).replace(tzinfo=None) ] = \
                                                ( -999999.0, 0) # 0 is the quality
                         else:
                            dataquality = self.calculateDataQuality(\
                                                float( value ) * unitConvertToM3perSec,\
                                                qualifier1, qualifier2 )
                            #logging.info( "qualifier1=%s qualifier2=%s dataquality=%s",qualifier1,qualifier2,dataquality )
                            self._timeValueQuality[ dateutil.parser.parse( valuetime )\
                                                .astimezone(pytz.utc).replace(tzinfo=None) ] = \
                                                ( float( value ) * unitConvertToM3perSec, dataquality )
                  

               self.obvPeriod = self._obvPeriod
               self.stationName = self._stationName
               self.stationID = self._stationID
               self.unit = self._unit
               self.timeValueQuality = self._timeValueQuality

           except Exception as e:
               raise RuntimeError( "WARNING: parsing JSON error: " + str( e )\
                               + ": " + jsonfilename + " skipping ..." )

           

        def loadWaterML2(self, waterml2filename ):
           """
              Read real-time stream flow data from a given WaterML 2.0 file
              by parsing the WaterML 2.0 file

              Input: waterml2filename - the WaterML 2.0 filename
           """        
           try:
                obvwml = etree.parse( waterml2filename )
                root= obvwml.getroot()
                identifier = root.find(\
                        '{http://www.opengis.net/gml/3.2}identifier')
                #remove 'USGS.' from stationID
                self._stationID = identifier.text[5:]

                self._stationName = \
                    root.find('{http://www.opengis.net/gml/3.2}name').text
                self._generationTime = dateutil.parser.parse(\
                root.find('{http://www.opengis.net/waterml/2.0}metadata')\
                 .find('{http://www.opengis.net/waterml/2.0}DocumentMetadata')\
                 .find('{http://www.opengis.net/waterml/2.0}generationDate')\
                 .text ).astimezone(pytz.utc).replace(tzinfo=None)
                timePeriod = root.find(\
                '{http://www.opengis.net/waterml/2.0}observationMember')\
                .find('{http://www.opengis.net/om/2.0}OM_Observation')\
                .find( '{http://www.opengis.net/om/2.0}phenomenonTime')\
                .find('{http://www.opengis.net/gml/3.2}TimePeriod')
                beginTime = timePeriod.find(\
                        '{http://www.opengis.net/gml/3.2}beginPosition').text
                endTime = timePeriod.find(\
                        '{http://www.opengis.net/gml/3.2}endPosition').text
                self._obvPeriod         = dateutil.parser.parse( beginTime )\
                                  .astimezone(pytz.utc).replace(tzinfo=None), \
                                dateutil.parser.parse( endTime )\
                                  .astimezone(pytz.utc).replace(tzinfo=None)

                uom = root.find(\
                '{http://www.opengis.net/waterml/2.0}observationMember')\
                .find('{http://www.opengis.net/om/2.0}OM_Observation')\
                .find('{http://www.opengis.net/om/2.0}result')\
                .find('{http://www.opengis.net/waterml/2.0}MeasurementTimeseries')\
                .find('{http://www.opengis.net/waterml/2.0}defaultPointMetadata')\
                .find('{http://www.opengis.net/waterml/2.0}DefaultTVPMeasurementMetadata')\
                .find('{http://www.opengis.net/waterml/2.0}uom')

                self._unit = uom.get( '{http://www.w3.org/1999/xlink}title')

                if self._unit == 'ft3/s':
                  unitConvertToM3perSec = 0.028317
                elif self._unit == 'm3/s':
                  unitConvertToM3perSec = 1.0
                else:
                  raise RuntimeError( "FATAL ERROR: Unit " + self._unit + \
                                   " is not known. ")
                  
                self._unit = 'm3/s'

                measurementTS = root.find(\
                '{http://www.opengis.net/waterml/2.0}observationMember')\
                .find('{http://www.opengis.net/om/2.0}OM_Observation')\
                .find('{http://www.opengis.net/om/2.0}result')\
                .find('{http://www.opengis.net/waterml/2.0}MeasurementTimeseries')

                qualifier1 = measurementTS.find( \
                   '{http://www.opengis.net/waterml/2.0}defaultPointMetadata')\
                   .find( \
                   '{http://www.opengis.net/waterml/2.0}DefaultTVPMeasurementMetadata')\
                   .find( \
                   '{http://www.opengis.net/waterml/2.0}qualifier')\
                   .find( \
                   '{http://www.opengis.net/swe/2.0}Category')\
                   .find( \
                   '{http://www.opengis.net/swe/2.0}value').text

                self._timeValueQuality = dict()

                for point in measurementTS.findall(\
                      '{http://www.opengis.net/waterml/2.0}point'):
                   valuetime = point.find(\
                      '{http://www.opengis.net/waterml/2.0}MeasurementTVP')\
                     .find('{http://www.opengis.net/waterml/2.0}time')
                   if valuetime is not None:
                     value = point.find(\
                      '{http://www.opengis.net/waterml/2.0}MeasurementTVP')\
                       .find('{http://www.opengis.net/waterml/2.0}value')

                     pointmetadata = point.find(\
                      '{http://www.opengis.net/waterml/2.0}MeasurementTVP')\
                       .find('{http://www.opengis.net/waterml/2.0}metadata')

                     qualifier2 = ''
                     if pointmetadata is not None:
                        measurementMetadataQualifiers = pointmetadata.find(\
                        '{http://www.opengis.net/waterml/2.0}TVPMeasurementMetadata')\
                        .findall('{http://www.opengis.net/waterml/2.0}qualifier')

                        if len( measurementMetadataQualifiers ) > 0:
                          qualifier1 = measurementMetadataQualifiers[ 0 ].find(\
                            '{http://www.opengis.net/swe/2.0}Category')\
                            .find( \
                              '{http://www.opengis.net/swe/2.0}value').text

                          if len( measurementMetadataQualifiers ) > 1:
                              qualifier2 = measurementMetadataQualifiers[ 1 ].find(\
                                '{http://www.opengis.net/swe/2.0}Category')\
                                 .find( \
                                   '{http://www.opengis.net/swe/2.0}value').text

                     isValueNull = value.get(\
                         '{http://www.w3.org/2001/XMLSchema-instance}nil') 

                     if isValueNull:
                       if isValueNull == "true":
                          self._timeValueQuality[ dateutil.parser.parse( valuetime.text )\
                                       .astimezone(pytz.utc).replace(tzinfo=None)] = \
                                ( -999999.0, 0) # 0 is the quality
                       else:

                          dataquality = self.calculateDataQuality(\
                                   float( value.text ) * unitConvertToM3perSec,\
                                    qualifier1, qualifier2 )

                          #logging.info( "111: qualifier1=%s qualifier2=%s dataquality=%s",qualifier1,qualifier2,dataquality )
                          self._timeValueQuality[ dateutil.parser.parse( valuetime.text )\
                                       .astimezone(pytz.utc).replace(tzinfo=None)] = \
                                ( float( value.text ) * unitConvertToM3perSec,\
                                  dataquality ) 
                     else:

                        dataquality = self.calculateDataQuality(\
                                   float( value.text ) * unitConvertToM3perSec,\
                                   qualifier1, qualifier2 )
                        #logging.info( "222: qualifier1=%s qualifier2=%s dataquality=%s",qualifier1,qualifier2,dataquality )
                        self._timeValueQuality[ dateutil.parser.parse( valuetime.text )\
                                     .astimezone(pytz.utc).replace(tzinfo=None)] = \
                                ( float( value.text ) * unitConvertToM3perSec,\
                                  dataquality ) 

           except Exception as e:
               raise RuntimeError( "WARNING: parsing water XML error: " + str( e )\
                               + ": " + waterml2filename + " skipping ..." )

           self.obvPeriod = self._obvPeriod
           self.stationName = self._stationName
           self.stationID = self._stationID
           self.unit = self._unit
           self.timeValueQuality = self._timeValueQuality

        def loadCSV(self, csvfilename ):
            """
              Read real-time stream flow data from a given CSV file
              by parsing the CSV file

              Input: csvfilename - the CSV filename
            """        

            self._timeValueQuality = dict()
            with open( csvfilename ) as csvfile:
                 reader = csv.DictReader( csvfile )
                 for row in reader:
                    dischstr = row['X_00060_00011']
                    discharge = float( dischstr ) * 0.028317 # cfs to cms
                    if row['X_00060_00011'] != '-999999' :

                       qualifiers =  row['X_00060_00011_cd'].split()
                       qualifier1 = qualifiers[ 0 ]
                       if len( qualifiers ) > 1 :
                         qualifier2 = qualifiers[ 1 ]
                       else:
                         qualifier2 = ''

                       dataquality = self.calculateDataQuality( \
                                  discharge, qualifier1, qualifier2 )
                       self._timeValueQuality[ datetime.strptime( \
                          row[ 'dateTime' ] + ' ' + row[ 'tz_cd' ], \
                          '%Y-%m-%d %H:%M:%S %Z' ) ] = \
                                    ( discharge, dataquality ) 
                    else:
                       self._timeValueQuality[ datetime.strptime( \
                          row[ 'dateTime' ] + ' ' + row[ 'tz_cd' ], \
                          '%Y-%m-%d %H:%M:%S %Z' ) ] = \
                                 ( -999999.0, 0 )

                 self._stationID = row[ 'site_no' ]

            timekeys = sorted( self._timeValueQuality.keys() )

            self._obvPeriod = timekeys[ 0 ], timekeys[ -1 ]
            self._stationName = self._stationID
            self._generationTime = timekeys[ 0 ]
            self._unit = 'm3/s'

            self.obvPeriod=self._obvPeriod
            self.stationName=self._stationID
            self.stationID=self._stationID
            self.unit=self._unit
            self.timeValueQuality=self._timeValueQuality

        def calculateDataQuality(self, value, qualifier1, qualifier2 ):
              """
              Calculate the real-time data quality for given  qualifier codes
              found in the WaterML 2.0 files.

              Input: value - the stream flow value
                     qualifier1 - the first qualifier
                     qualifier2 - the second qualifier

              Return: Data quality value
              """
#
#%STATUS_CODES =
#  (
#   '--'  => 'Parameter not determined',
#   '***' => 'Temporarily unavailable',
#   'Bkw' => 'Flow affected by backwater',
#   'Dis' => 'Data-collection discontinued',
#   'Dry' => 'Dry',
#   'Eqp' => 'Equipment malfunction',
#   'Fld' => 'Flood damage',
#   'Ice' => 'Ice affected',
#   'Mnt' => 'Maintenance in progress',
#   'Pr'  => 'Partial-record site',
#   'Rat' => 'Rating being developed or revised',
#   'Ssn' => 'Parameter monitored seasonally',
#   'ZFl' => 'Zero flow',
#  );
#
#Also "approval" codes applicable to both UV and DV data:
#
#%DATA_AGING_CODES =
#  (
#   'P' => ['0', 'P', 'Provisional data subject to revision.'],
#   'A' => ['0', 'A', 'Approved for publication -- Processing and review
#completed.'],
#  );
#
#
#There are also unit value quality codes the vast majority of which you should never see.  The ones in red below "may" be the only one you can expect to see in the public view:
#
#%UV_REMARKS =
#  (
#   # -- Threshold flags
#   'H' => ['1', 'H', 'Value exceeds "very high" threshold.'],
#   'h' => ['1', 'h', 'Value exceeds "high" threshold.'],
#   'l' => ['1', 'l', 'Value exceeds "low" threshold.'],
#   'L' => ['1', 'L', 'Value exceeds "very low" threshold.'],
#   'I' => ['1', 'I', 'Value exceeds "very rapid increase" threshold.'],
#   'i' => ['1', 'i', 'Value exceeds "rapid increase" threshold.'],
#   'd' => ['1', 'd', 'Value exceeds "rapid decrease" threshold.'],
#   'D' => ['1', 'D', 'Value exceeds "very rapid decrease" threshold.'],
#   'T' => ['1', 'T', 'Value exceeds "standard difference" threshold.'],
#
#   # -- Source Flags
#   'o' => ['1', 'o', 'Value was observed in the field.'],
#   'a' => ['1', 'a', 'Value is from paper tape.'],
#   's' => ['1', 's', 'Value is from a DCP.'],
#   '~' => ['1', '~', 'Value is a system interpolated value.'],
#   'g' => ['1', 'g', 'Value recorded by data logger.'],
#   'c' => ['1', 'c', 'Value recorded on strip chart.'],
#   'p' => ['1', 'p', 'Value received by telephone transmission.'],
#   'r' => ['1', 'r', 'Value received by radio transmission.'],
#   'f' => ['1', 'f', 'Value received by machine readable file.'],
#   'z' => ['1', 'z', 'Value received from backup recorder.'],
#
#   # -- Processing Status Flags
#   '*' => ['1', '*', 'Value was edited by USGS personnel.'],
#
#   # -- Other Flags (historical UVs only)
#   'S' => ['1', 'S', 'Value could be a result of a stuck recording instrument.'],
#   'M' => ['1', 'M', 'Redundant satellite value doesnt match original value.'],
#   'Q' => ['1', 'Q', 'Value was a satellite random transmission.'],
#   'V' => ['1', 'V', 'Value was an alert value.'],
#
#   # -- UV remarks
#   '&' => ['1', '&', 'Value was computed from affected unit values by unspecified reasons.'],
#   '<' => ['0', '<', 'Actual value is known to be less than reported value.'],
#   '>' => ['0', '>', 'Actual value is known to be greater than reported value.'],
#   'C' => ['1', 'C', 'Value is affected by ice at the measurement site.'],
#   'B' => ['1', 'B', 'Value is affected by backwater at the measurement site.'],
#   'E' => ['0', 'e', 'Value was computed from estimated unit values.'],
#   'e' => ['0', 'e', 'Value has been estimated.'],
#   'F' => ['1', 'F', 'Value was modified due to automated filtering.'],
#   'K' => ['1', 'K', 'Value is affected by instrument calibration drift.'],
#   'R' => ['1', 'R', 'Rating is undefined for this value.'],
#   'X' => ['1', 'X', 'Value is erroneous and will not be used.'],
#  );
#
#Some daily value specific codes:
#
#%DV_REMARKS =
#  (
#   '&' => ['1', '&',  'Value was computed from affected unit values by unspecified reasons.'],
#   '1' => ['1', '1',  'Daily value is write protected without any remark code to be printed.'],
#   '2' => ['1', '2',  'Remark is suppressed and will not print on a publication daily values table.'],
#   '<' => ['0', '<', 'Actual value is known to be less than reported value.'],
#   '>' => ['0', '>', 'Actual value is known to be greater than reported value.'],
#   'E' => ['0', 'e', 'Value was computed from estimated unit values.'],
#   'e' => ['0', 'e', 'Value has been estimated.'],
#  );
#
#I think we should set the ?e? (estimated) to 50% since we don?t know how the estimation was done, set the ?&? to 25%, and may be the ?*? to 5% since the data could be incomplete. The remainder set to 0 for the time being. That should wrap up this QC at this stage. We can always revisit them in the future as we progress.
#
#I made contact today with the USGS staff on location here and inform them that I will visit them next week for question about the rating curve gages. May be we can visit them together.
#
# 
#
#Cheers,
#
# 
#
#Oubeid
#
#Let?s stick with a weight of ?0? then for ?***?. I don?t have a lot of confidence on the data appended with that code.
#
# 
#
#Cheers,
#
# 
#
#Oubeid
#
#  
#Date: Wed, 18 Nov 2015 14:41:13 -0700
#From: James McCreight <jamesmcc@ucar.edu>
#To: Zhengtao Cui - NOAA Affiliate <zhengtao.cui@noaa.gov>,
#    David Gochis <gochis@ucar.edu>,
#    Brian Cosgrove - NOAA Federal <brian.cosgrove@noaa.gov>
#Subject: preprocessing followup
#Parts/Attachments:
#   1   OK     33 lines  Text (charset: UTF-8)
#   2 Shown   ~28 lines  Text (charset: UTF-8)
#----------------------------------------
#
#Zhengtao-Great to talk to you about the code today.  
#Here's the recap of things to do:
# *  unit conversion to cms for the csv code
# *  set quality to zero instead of raising errors in the quality code
# *  use an upper limit for flow: 90,000 cms (see note below).
# *  exclude 0 (?) as a valid flow
#Some notes on the last two bullets:
### JLM limit maximum flows to approx twice what I believe is the largest
#gaged flow
### on MS river *2 
### http://nwis.waterdata.usgs.gov/nwis/peak?site_no=07374000&agency_cd=USGS&format=h
#tml [nwis.waterdata.usgs.gov] 
### baton rouge 1945: 1,473,000cfs=41,711cms
### multiply it roughly by 2
#dataDf$quality[which(dataDf$discharge.cms > 0 & dataDf$discharge.cms <
#90000)] <- 100
#
#As the above shows, I'm really torn about using zero values from NWIS. I'm
#inclined not to until we develop more advanced QC procedures. 
#
#Thanks,
#
#James

              quality = 0

              if value  <= 0 or value > 90000 : # see James' email on Nov. 18 above
                 quality = 0
              else :
                  if ( qualifier2  and len( qualifier2 ) > 0 ) :
                     if qualifier1 == 'A' or qualifier1 == 'P' :
                         if  qualifier2 == 'e' :
                             quality = 50
                         elif qualifier2 == '&' :
                             quality = 25
                         elif qualifier2 == '*' :
                             quality = 5
                         elif qualifier2 == 'A' :
                             quality = 100 
                         elif qualifier2 == 'P' :
                             quality = 100 
                         else:
                             quality = 0
#                     else :
#                        raise RuntimeError( "ERROR: unknow qualifier: " + \
#                                qualifier1 )
                  else:
                      if qualifier1 == 'A' or qualifier1 == 'P' :
                           quality = 100
#                      else:
#                        raise RuntimeError( "ERROR: unknow qualifier: " + \
#                                qualiifer1 )
              return quality             
