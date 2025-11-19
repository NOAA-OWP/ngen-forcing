#!/usr/bin/env python
###############################################################################
#  File name: parallel_download_master.py
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  7/12/2017                                         #
#                                                                             #
#  Description: The master process that connect the USGS real-time flow data  #
#               server and download real-time data for each HUC using         #
#               three cocurrent process.                                      #
#                                                                             #
###############################################################################

import sys, time, os
import multiprocessing
import find_changed_site_for_huc
import usgs_iv_retrieval

if __name__ == "__main__":
   odir = usgs_iv_retrieval.main(sys.argv[1:])

#
# Initialize the list of HUCs and process
#
proc_hucs = dict( [ ('2604', [2, 17, 12, 18, 8, 13, 20 ] ), \
                    ('2634', [10, 5, 4, 1, 16, 6, 19 ] ), \
                    ('2920', [3, 7, 11, 15, 14, 9, 21 ] ) ] )

#
# Start the downloading process
# the total number of stations for the above HUCs is 2604
#
procs= dict()
for proc, hucs in iter( proc_hucs.items() ):
  procs[ proc] =  multiprocessing.Process( name= proc,\
                target=usgs_iv_retrieval.download_for_hucs,\
                args=( odir, hucs, proc ) ) 

for p in procs.values():
  print( 'start : ', p.name )
  p.start()

while True:

  time.sleep( 2*3600 )

  proc_restarted = []
  for p in procs.values():
      if p.is_alive():
            if os.path.isfile( odir + '/usgs_iv_retrieval_' + p.name ):
                  t = os.stat(  odir + '/usgs_iv_retrieval_' + p.name )
                  c = t.st_mtime
                  if c < time.time() - 180 :
                    print( 'process: ', p.name, 'stalled!!')
                    # restart
                    p.terminate()   
                    pr = multiprocessing.Process( name= p.name,\
                         target=usgs_iv_retrieval.download_for_hucs,\
                         args=( odir, proc_hucs[ p.name ], p.name ) ) 
                    proc_restarted.append( pr )
                    pr.start()
                    print( 'restarted : ', p.name )
            else:
                    print( 'process: ', p.name, 'status file doesn\'t exist')
                    p.terminate()   
                    pr = multiprocessing.Process( name= p.name,\
                         target=usgs_iv_retrieval.download_for_hucs,\
                         args=( odir, proc_hucs[ p.name ], p.name ) ) 
                    proc_restarted.append( pr )
                    pr.start()
                    print( 'process: ', p.name, 'restarted :' )
      else:
                    
          print( 'process: ', p.name, ' is not alive.')
          p.terminate()   
          pr = multiprocessing.Process( name= p.name,\
              target=usgs_iv_retrieval.download_for_hucs,\
              args=( odir, proc_hucs[ p.name ], p.name ) ) 
          proc_restarted.append( pr )
          print( 'process: ', p.name, 'restarted :')
          pr.start()

  if proc_restarted:
      for p in proc_restarted:
          procs[ p.name ] = p

for p in procs.values():
  p.join()
  print( '%s.exitcode = %s' % (p.name, p.exitcode) )

