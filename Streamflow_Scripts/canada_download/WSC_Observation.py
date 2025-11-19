import os, sys, time, csv
from string import *
from datetime import datetime, timedelta
# import xml.utils.iso8601
import xml.etree.ElementTree as etree
from TimeSliceC import TimeSliceC
import pytz
from EmptyDirOrFileException import EmptyDirOrFileException

class WSC_Observation:
    "Store one WSC data set "
    
    def __init__(self, csvfilename):
        self.timevalue = dict()
        
        #
        #  Define the column names I want to use
        #  If WSC changes file formats, this may need to be updated.
        #
        #csvfields = ('id', 'date', 'level', 'lgrade', 'lsymbol', 'lqaqc', 
        #             'discharge', 'dgrade', 'dsymbol', 'dqaqc')
        #
        #The header has changed since Feb 22, 2023.
        csvfields = ('ID', 'Date', 'Water Level / Niveau d\'eau (m)',
                'Grade', 'Symbol / Symbole', 'QA/QC', 
                     'Discharge / DÃit (cms)', 'Grade', 'Symbol / Symbole', 'QA/QC')
        #
        #Using utf-8 causes error like this,
        #'utf-8' codec can't decode byte 0xe9 in position 81: invalid continuation byte
        #use 'latin-1' or 'ISO-8859-1' instead
        #
        #with open(csvfilename, 'r', encoding="utf-8") as csvfile:
        #with open(csvfilename, 'r', encoding="latin-1") as csvfile:
        with open(csvfilename, 'r', encoding="ISO-8859-1") as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=csvfields)
            next(reader)
            for row in reader:
                try:
                    dischstr = row['Discharge / DÃit (cms)']
                    discharge = float(dischstr) * 1.0
                    qstr = row['QA/QC']
                except:
                    discharge = -999999.0
                dkey = None
                try:
                    #
                    #  Split the date string into two parts... Y-M-D and H:M:S
                    #  Then parse them into a timezone-naive object
                    #  Per the documentation from WSC, this timestamp is described as:
                    #  "data timestamp in ISO 8601 format, Local Standard Time (LST)"
                    #  So if the timestamp is
                    #    2018-08-16T13:30:00-0500
                    #  I am interpreting that as 1:30 pm local time, and that local time
                    #  is 5 hours behind UTC. If this interpretation is incorrect, the
                    #  following code should be modified appropriately.
                    #
                    #  Once I have a datetime object representing the local time, I can
                    #  use the timezone info to translate that into UTC, which is what
                    #  will be used as the data key value.
                    #
                    #  NOTE!!!!!  When storing the data, we are rounding/stripping the
                    #  seconds field to 0.  So I need to do the same thing here.
                    #  That is accomplished very simply, by replacing whatever was 
                    #  specified for seconds with '00'.  If this is not done we end up
                    #  with mismatch issues when we are merging timeslices.
                    #
                    s = row['Date'].split("T")
                    ymd = s[0]
                    hms = s[1][:6] + '00'
                    dsnz = ymd + ' ' + hms        # format is YYYY-MM-DD HH:MM:SS
                    dtnz = datetime.strptime(dsnz, '%Y-%m-%d %H:%M:%S')    # timezone-naive object

                    tz = s[1][8:]
                    tzh = int(tz[1:3])      # time zone hours, *absolute value*; typically 4 or 5 for GL region
                    tzm = int(tz[4:])       # time zone minutes, typically 0 for GL region
                    tzoff = timedelta(hours=tzh, minutes=tzm)
                    
                    #
                    #  Adjust the time appropriately so that it represents
                    #  the UTC time.
                    #
                    if (tz[0] == '-'): 
                       dtu = dtnz + tzoff     # this is GL case
                    else:
                       dtu = dtnz - tzoff
                    
                    #
                    #  Make it timezone-aware, assigning it to UTC timezone and
                    #  then using that datetime value as the dictionary key
                    #
                    dkey = dtu.replace(tzinfo=pytz.UTC)
                except Exception as e:
                    raise Exception( \
                             'cannot parse date... {}'.format(e) ) from e
                if dkey and discharge >= 0:
                    dataquality = self.calculateDataQuality(discharge, \
                                    int( qstr ) )
                    self.timevalue[dkey] = (discharge, dataquality) 
                else:
                    self.timevalue[dkey] = (-999999.0, 0)
                self.stationID = 'CAN.' + row['ID']

        timekeys = sorted( self.timevalue.keys() )

        self.obvPeriod = timekeys[0], timekeys[-1]
        self.stationName = self.stationID
        self.generationTime = timekeys[0]
        self.unit = 'm3/s'


    def calculateDataQuality(self, value, qacode):
        """  Calculate a quality code for the discharge measurement based on
        the value and associated QA/QC code.  The defined QA/QC codes are:
        1 = preliminary, 2 = reviewed, 3 = checked, 4 = approved 
        """
########################################################################
#Date: Fri, 10 Jul 2020 12:49:57 -0600
#From: James McCreight <jamesmcc@ucar.edu>
#To: Zhengtao Cui <zhengtao.cui@noaa.gov>
#Cc: Tim Hunter - NOAA Federal <tim.hunter@noaa.gov>,
#Brian Cosgrove - NOAA Federal <brian.cosgrove@noaa.gov>,
#Arezoo RafieeiNasab <arezoo@ucar.edu>, David Gochis <gochis@ucar.edu>,
#Aubrey Dugger <adugger@ucar.edu>, Ryan Cabell <rcabell@ucar.edu>,
#Laura Read <lread@ucar.edu>
#Subject: Re: High priority....Canada streamflow issue
#Parts/Attachments:
#1   OK    ~342 lines  Text (charset: UTF-8)
#2 Shown   ~728 lines  Text (charset: UTF-8)
#----------------------------------------
#So, I would say it depends on our expectations of what the codes really
#mean, but the weight penalties should be thought of as "relative to model
#error" with .5 giving something like equal weight. Generally, the
#observation is going to be much better (there are exceptions) even when it's
#not revised (at least for USGS).... unless this is just not true for these
#gages.
#I might suggest the weighting scheme below in code, based on my best guess.
#Here is my basic thought process:
#Within 28 hours (the extended look back period), we dont get to QC levels 3
#and 4 because that process likely takes longer.
#So, once we reach 2, we are probably doing as well as can be expected.
#It would be informative to see how much the OPERATIONAL data stream has
#quality 1 and 2 and what the ?tis between obs time and time that that flag
#arrives.
#It might be informative to also see if/how much the discharge values change
#when their quality codes change.
#It is possible that the necessary data/flow is beyond what is currently
#setup, but doing an aggressive saving of the canada gage obs files could
#shed some light.
#
#def calculateDataQuality(self, value, qacode):
#   """  Calculate a quality code for the discharge measurement based on
#     the value and associated QA/QC code.  The defined QA/QC codes are:
#     1 = preliminary, 2 = reviewed, 3 = checked, 4 = approved
#   """
#   quality = 0
#   if value <= 0 or value > 9000:
#      quality = 0
#   else:
#      if qacode == 1:
#         quality = 75
#      elif qacode == 2:
#         quality = 100
#      elif qacode == 3:
#         quality = 100
#      elif qacode == 4:
#         quality = 100
#      else:
#         quality = 0
#   return quality
#-------------------------------------------------------
#James L. McCreight
#NCAR Research Applications Lab
#office:       FL2 2065
#office phone: 303-497-8404
#cell:         831-261-5149
########################################################################
#Date: Thu, 16 Jul 2020 14:15:12 -0400
#From: Brian Cosgrove - NOAA Federal <brian.cosgrove@noaa.gov>
#To: Arezoo RafieeiNasab <arezoo@ucar.edu>
#Cc: Zhengtao Cui <zhengtao.cui@noaa.gov>,
#    James McCreight <jamesmcc@ucar.edu>,
#    Tim Hunter - NOAA Federal <tim.hunter@noaa.gov>,
#    David Gochis <gochis@ucar.edu>, Aubrey Dugger <adugger@ucar.edu>,
#    Ryan Cabell <rcabell@ucar.edu>, Laura Read <lread@ucar.edu>
#    Subject: Re: High priority....Canada streamflow issue
#    Parts/Attachments:
#       1.1   OK     509 lines  Text (charset: UTF-8)
#       1.2 Shown    ~39 KB     Text (charset: UTF-8)
#       2     OK     977 KB     Image
#----------------------------------------
#Okay, barring any further input/concerns from folks, let's do this....
#Zhengtao, can you alter it so that it gives 100% weight (quality) to the
#incoming obs even if the QC code is only '1'?  I don't' think we're seeing
#the desired impact of the obs currently....let's see how it looks after that
#change in real-time.
#
#So this would change to quality = 100
#
#  >             if qacode == 1:
#  >                 quality = 75  
#
#  Thanks,
#  Brian
#
        quality = 0
        if value <= 0 or value > 9000: 
            quality = 0
        else:
            if qacode == 1:
                quality = 100 
            elif qacode == 2:
                quality = 100 
            elif qacode == 3:
                quality = 100 
            elif qacode == 4:
                quality = 100
            else:
                quality = 0
        return quality             

    def getTimeValueAt(self, at_time, resolution = timedelta() ):
        closestTimes = []
        distances = []
        for k in sorted(self.timevalue):
            td = abs(k - at_time)
            if td <= resolution/2:
                closestTimes.append(k)
                distances.append(td)
        if closestTimes:
            closest = [x for y,x in sorted(zip(distances,closestTimes))] [0]
            return (closest, self.timevalue[closest])
        else:
            return None
            

class All_WSC_Observations:
    """Store all WSC data in a given directory"""
    def __init__(self, wscdatadir ):
        self.source = wscdatadir
        self.wscobvs = []
        if not os.path.isdir(wscdatadir):
            raise RuntimeError("FATAL ERROR: " + wscdatadir 
                  + " is not a directory or does not exist. ")

        twodaysago = datetime.now() - timedelta( days = 2 )
        for file in os.listdir(wscdatadir): 
          if file.endswith(".csv"):
             fname = wscdatadir + '/' + file
             st = os.stat( fname )
             #
             # Don't process data older than 2 days.             
             #
             if datetime.fromtimestamp( st.st_mtime) > twodaysago:
                try:
                    print( 'Reading ' + fname + ' ... ' )
                    self.wscobvs.append(WSC_Observation(fname))
                except Exception as e:
                    print( "WARNING: parsing CSV file error: " +  \
                         file + ", skipping ... " )
                    print( e )
                    continue

        if not self.wscobvs:
            raise EmptyDirOrFileException( "Input directory " + wscdatadir \
                  + " has no Water Survey of Canada CSV files or the "
                  "Water Survey CSV files contain no flow data!" )

            self.index = -1

        self.timePeriod = self.wscobvs[0].obvPeriod
        for obv in self.wscobvs:
            if self.timePeriod[0] > obv.obvPeriod[0]:
                self.timePeriod = ( obv.obvPeriod[0], self.timePeriod[1])
            if self.timePeriod[1] < obv.obvPeriod[1]:
                self.timePeriod = ( self.timePeriod[0], obv.obvPeriod[1])


    def __iter__(self):
        return self

    def next(self):
        if self.index == len( self.wscobvs ) - 1:
            self.index = -1
            raise StopIteration
        self.index = self.index + 1
        return self.wscobvs[self.index]

    def timePeriodForAll(self):
        return self.timePeriod

    #
    #  Step through the observations, finding the entry that
    #  is closest (time-wise) to the desired timestamp. 
    #  Append that observation to the station_time_value_list.
    #  Then create a TimeSliceC object from all of the observations in that list.
    #  
    def makeTimeSlice(self, timestamp, timeresolution):
        station_time_value_list = []
        for obv in self.wscobvs:
            closestObv = obv.getTimeValueAt( timestamp, timeresolution ) 
            if closestObv:
                c0  = closestObv[0]
                c10 = closestObv[1][0]
                c11 = closestObv[1][1]
                station_time_value_list.append((obv.stationID, c0, c10, c11)) 
        timeSlice = TimeSliceC(timestamp, timeresolution, station_time_value_list)
        return timeSlice

    def makeAllTimeSlices(self, timeresolution, outdir):
        #
        # the time resolutions must divide 60 minutes with no remainder
        #
        if 3600 % timeresolution.seconds != 0:
            raise RuntimeError( "FATAL Error: Time slice resolution must "
                                "divide 60 minutes with no remainder." )

        #
        # Always start two days ago because sometimes the downloaded realtime
        # files contain old data older than six months, See the email from NCO
        # Simon Hsiao on Dec 16, 2022, with subject 
        # "20221216 12z nwm_canada_timeslices job hung".
        #
        #
#        NWM team,
#
#        The 20221216 12z nwm_canada_timeslices job hung failed seeing hung reached walltime 10 min. here are the job logfiles and working dir for your investigations, rerun still hung  - 
#        /lfs/h1/ops/prod/output/20221216/nwm_canada_timeslices_12_1725.o33696851  -- 1st run
#        /lfs/h1/ops/prod/output/20221216/nwm_canada_timeslices_12_1739.o33703661  --  2nd run
#        /lfs/f1/ops/prod/tmp/nwm_canada_timeslices_12_1725.33696851.dbqs01   - 1st run working dir
#        /lfs/f1/ops/prod/tmp/nwm_canada_timeslices_12_1739.33703661.dbqs01/   -- 2nd run working dir
#        In the working dir,there are some 2022-06-01 15 min WscTimeSlice.ncdf data files as below , why back to 202206 ?
#        2022-06-01_05_00_00.15min.wscTimeSlice.ncdf
#        ...
#        2022-06-03_23_00_00.15min.wscTimeSlice.ncdf
#
#        Thanks,
#
#        /Simon
#        SPA Office

        #
        # need to use UTC time here
        startTime = datetime.now().replace(tzinfo=pytz.UTC) - timedelta( days = 2 )

        #Round the start time to the nearist 0, 15, 30 and 45 minutes
        #Otherwise the filename will be wrong.

        #startTime = self.timePeriod[0]

        if startTime.minute >= 8  and startTime.minute <= 22:
            startTime = datetime( startTime.year, startTime.month, 
                                           startTime.day, startTime.hour, 
                                           15, tzinfo=startTime.tzinfo )
        elif startTime.minute >= 23  and startTime.minute <= 37:
            startTime = datetime( startTime.year, startTime.month, 
                                           startTime.day, startTime.hour, 
                                           30, tzinfo=startTime.tzinfo )
        elif startTime.minute >= 38  and startTime.minute <= 52:
            startTime = datetime( startTime.year, startTime.month, 
                                           startTime.day, startTime.hour, 
                                           45, tzinfo=startTime.tzinfo )
        elif startTime.minute < 8: 
            startTime = datetime( startTime.year, startTime.month, 
                                           startTime.day, startTime.hour, 
                                           0, tzinfo=startTime.tzinfo )
        else: # > 52
            startTime = datetime( startTime.year, startTime.month, 
                                           startTime.day, startTime.hour, 
                                           0, tzinfo=startTime.tzinfo ) + \
                                           timedelta( hours = 24 )

        while startTime < self.timePeriod[0]:
            startTime += timeresolution

        if startTime > self.timePeriod[1]: 
            raise RuntimeError("FATAL Error: observation time period wrong!")

        count = 0
        print ("Timeslice start time: ", startTime )
        while startTime <= self.timePeriod[1]:
            print ("making time slice for ", startTime.isoformat())
            oneSlice = self.makeTimeSlice(startTime, timeresolution)
            if ( not oneSlice.isEmpty() ):
                updatedOrNew = True
                slicefilename = outdir + '/' +oneSlice.getSliceNCFileName() 
                if os.path.isfile( slicefilename ):
                    oldslice = TimeSliceC.fromNetCDF( slicefilename )
                    updatedOrNew = oneSlice.mergeOld( oldslice )
                if updatedOrNew:
                    oneSlice.toNetCDF(outdir)
                    print (oneSlice.getSliceNCFileName() + " updated!")
                else:
                    print (oneSlice.getSliceNCFileName() + " not updated!")
                count = count + 1

            startTime += timeresolution
        return count
