#!/usr/bin/env python

from contextlib import closing
import os, sys, time, getopt, re
import datetime, time
import pytz
from string import *
from six.moves import urllib
import ssl
#import HTMLParser


def main(argv):
   """
       Function to get output directory

       Return: Output directory
   """
   outputdir = ''
   try:
      opts, args = getopt.getopt(argv,"h:o:",["odir="])
   except getopt.GetoptError:
      print( 'canadian_flow_retrieval.py -o <outputdir>' )
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print( 'canadian_flow_retrieval.py -o <outputdir>' )
         sys.exit()
      elif opt in ("-o", "--odir"):
         outputdir = arg
         if not os.path.exists( outputdir ):
             os.makedirs( arg )
  
   print( 'Output dir is "', outputdir )
   return outputdir
#-----------------------------------------------------------        
#
# Download real time stream flow data from the Canadian "datamart" 
# for a given list of stations.
#
def fetch_ca_sites( sites, province, odir ):

    good_sites = ()
    fail_sites = ()
    #
    # Loop through each station
    #
    for site_id in sites:
        print( datetime.datetime.now(), end = " --- " )
        print( 'downloading ', site_id )
        #
        # Construct the query URL
        # 
        dirstring = 'https://dd.weather.gc.ca/hydrometric/csv/' + province + '/hourly/'
        filename = province + '_' + site_id + '_hourly_hydrometric.csv'
        URL = ( dirstring + filename )
        localFile = odir + os.path.sep + filename
        print( datetime.datetime.now(), end = " --- " )
        print( "URL: " + URL)
        print( datetime.datetime.now(), end = " --- " )
        print( "local file: " + localFile)
               
        #
        # Connect to the server
        # 
        try:
            lFile = urllib.request.urlretrieve(URL, localFile)
            good_sites = good_sites + (site_id,)
        except IOError as e:
            #print ('WARN: site : ', site_id, ' skipped - ' #, e.reason)
            #
            # If failed, remember the station id and continue
            # 
            fail_sites = fail_sites + ( site_id, )

    return good_sites, fail_sites

#-----------------------------------------------------------        
def build_download_list( province, odir ):
    #
    #  province = 'ON' or 'QC'  (Ontario or Quebec)
    #  odir = output directory
    #
    #
    #  Get the UTC offset, in seconds, then convert that to a
    #  timedelta object.
    #
    tz_offset = datetime.timedelta(seconds=time.timezone)

    #
    #  Define the regular expression search patterns that will be repeatedly used
    #  with the information on the web page listing.
    #    site_pattern will identify sites that are in the Great Lakes basin only ("02" is the key)
    #    ts_pattern is used for parsing the modification timestamp
    #
    site_pattern = re.compile(province + '_02' + '.{5}' + '_hourly_hydrometric.csv')
    ts_pattern   = re.compile('[\d]{4}-[\d]{2}-[\d]{2} [\d]{2}:[\d]{2}')

    #
    #  Define some useful date constants
    #
    missing_time = datetime.datetime(1000, 1, 1, 1, 1, 1, 1)
    future_offset = datetime.timedelta(days=99999)
    
    #
    #  Get the directory listing from the web page.
    #  Parse each line. If it is a file entry, check the "modified time" vs the
    #  local file's timestamp.  If the modified time is more recent, then we
    #  need to update that file, so add it to the list.
    #
    total_sites = 0
    sitelist=list()
    
    urllib.request.urlcleanup()
    urlstring = 'https://dd.weather.gc.ca/hydrometric/csv/' + province + '/hourly/'
    print( datetime.datetime.now(), end = " --- " )
    print( urlstring )
    with closing(urllib.request.urlopen(urlstring)) as dirlisting:
        for line in dirlisting:
            s = line.decode('ascii')
            match_site = site_pattern.search(s)
            if (match_site):
                total_sites += 1
                fname = match_site.group(0)       # e.g. "ON_02AB006_hourly_hydrometric.csv"
                id = fname[3:10]
               
                #
                #  Get timestamp for the existing local file with that same id (if
                #  it exists.)  Adjust to UTC, because the modification times reported 
                #  on the remote server are given in UTC.
                #  If the remote file's timestamp is newer, that means it has been
                #  updated since the last time we downloaded a file for that site.
                #
                local_file = odir + os.path.sep + fname
                l_mod_time = missing_time                     # default invalid value, long ago
                if os.path.isfile(local_file):
                    try:
                        lfm = os.path.getmtime(local_file)    # local file mod time (float value)
                        l_mod_time = datetime.datetime.fromtimestamp(lfm)
                        l_mod_time = l_mod_time + tz_offset
                    except:
                        l_mod_time = missing_time

                #
                #  Extract the modified time for the remote file (from the html dir listing)
                #  Format (in testing, at least) is, e.g. "2018-08-16 18:46".
                #  These are UTC times.
                #
                #  If the time retrieval fails (or formatting has changed since this
                #  code was updated), then the remote file is assigned a modification 
                #  date far into the future, which will force the file to be flagged as
                #  "new", and needing to be updated. The resulting sitelist will contain
                #  a list of the site IDs for those files that need to be updated.
                #
                match_ts = ts_pattern.search(s)
                if match_ts:
                    ds = match_ts.group(0)
                    try:
                        r_mod_time = datetime.datetime.strptime(ds, '%Y-%m-%d %H:%M')
                    except:
                        r_mod_time = datetime.datetime.now()
                else:
                    r_mod_time = l_mod_time + future_offset              # default future value

                #
                #  Compare timestamps and add to list if needed
                #
                try:
                    if (r_mod_time > l_mod_time): sitelist.append(id)
                except Exception:
                    pass
  
    print( datetime.datetime.now(), end = " --- " )
    print(len(sitelist), ' of ', total_sites, ' need to be updated')
    return sitelist
    
    
#-----------------------------------------------------------        
#
# The main function to download real time stream flow
#  
#
def canadian_flow_retrieval( odir ):

    #
    # This fixes the certificate verify error
    # ERROR:  [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:777)
    #
    try:
       _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
       # Legacy Python that doesn't verify HTTPS certificates by default
       pass
    else:
       # Handle target environment that doesn't support HTTPS verification
       ssl._create_default_https_context = _create_unverified_https_context

    #
    #  How often to check for new files (in minutes).
    #  It appears (by simple observation) that the files are 
    #  updated about every 30 minutes. Not all files are updated 
    #  at the same time, so setting this to an interval of about 
    #  10 minutes seems like a reasonable value to me, but users
    #  should adjust to whatever they deem appropriate.
    #
    download_frequency = 10

    #
    #  Time stamp when the process starts
    #
    loop_start = time.time()
    
    #
    #  Infinite loop
    #  For testing, disable the "while True:" line and enable the
    #  two loop number lines.
    #
    lnum = 0
#    while True:
#    while lnum < 25:             
    while lnum < 1:             
        lnum = lnum + 1
        #
        #  Query the Canadian web site to build a list of files that have been updated 
        #  since the last time we updated them on the local filesystem.
        #  Total number of sites available is approximately 500, so this executes quickly.
        #
        updated_ON = build_download_list( 'ON', odir )
        updated_QC = build_download_list( 'QC', odir )
        list_of_updated_sites = updated_ON + updated_QC
        
        #
        #  How many files need to be updated?
        #
        upd_count = 0               
        if (list_of_updated_sites):
            upd_count = len(list_of_updated_sites)

        #
        #  If count > 0, then update the files in the list.
        #  Two lists are returned:
        #     good_sites = list of sites that were successfully updated
        #     fail_sites = list of sites that failed when trying to update them.
        #
        if (upd_count > 0):
            good_sites_on, fail_sites_on = fetch_ca_sites( updated_ON, 'ON', odir )
            good_sites_qc, fail_sites_qc = fetch_ca_sites( updated_QC, 'QC', odir )
            good_sites = good_sites_on + good_sites_qc
            fail_sites = fail_sites_on + fail_sites_qc
            try:
                i = len(good_sites)
            except:
                i = 0
            try:
                j = len(fail_sites)
            except:
                j = 0
            print( datetime.datetime.now(), end = " --- " )
            print('There were ', i, ' good downloads and ', j, ' failed downloads')
            

        #
        #  Compute the correct amount of time to sleep before
        #  doing this again.  
        #
        loop_end = time.time()
        elapsed  = loop_end - loop_start      # seconds
        sleep_time = (download_frequency * 60.0) - elapsed
        
        #
        #  If the specified sleep time has already elapsed because
        #  the download took longer than the specified time, then
        #  just sleep for an arbitrary 10 seconds in order to
        #  be sure that everything gets to a fully reset state.
        #
        if (sleep_time <= 10): sleep_time = 10
        
        #
        #  Sleep for the specified number of seconds.
        #
#        print('sleeping for ', sleep_time, ' seconds')
#        time.sleep( sleep_time )
        loop_start = time.time()

#-----------------------------------------------------------        
#-----------------------------------------------------------  
      
#MyDir = '/gpfs/hps3/ptmp/Zhengtao.Cui/CanDA/test1'
#MyDir = '/gpfs/hps3/ptmp/Zhengtao.Cui/wscxml2'

#canadian_flow_retrieval( odir )
