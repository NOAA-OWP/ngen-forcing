#!/usr/bin/env python

import sys, time, os
import multiprocessing
import canadian_flow_retrieval

if __name__ == "__main__":
    odir = canadian_flow_retrieval.main(sys.argv[1:])
    print( "odir=" + odir)
    canadian_flow_retrieval.canadian_flow_retrieval( odir )
    
#
# Start the downloading process
#    "procs['can']=" creates a new process (thread), 
#     but does not make it start executing.  So in this case we are 
#     creating a thread, but it does not begin executing yet.
#
#  Note that this code/method is overly complicated for the single
#  process that is being created here, but I am keeping things as
#  close as possible to Zhengtao Ciu's code/method.  He was making
#  3 processes.
#
procs=dict()
procs['can'] = multiprocessing.Process( name='can',  \
        target=canadian_flow_retrieval, args=(odir) ) 

#
# Here is where we actually start those new threads executing
#
#for p in procs.values():
#    print( 'start : ' + p.name )
#    p.start()

##
## Infinite loop to keep the process running.
## If it stalled or crashed, restart.
##
#while True:
#    time.sleep( 2*3600 )
#    proc_restarted = []
#    for p in procs.values():
#        if p.is_alive():
#            if os.path.isfile( odir + '/canadian_flow_retrieval'):
#                t = os.stat( odir + '/canadian_flow_retrieval')
#            c = t.st_mtime
#            if c < time.time() - 180 :
#                print( 'process: ', p.name, 'stalled!!' )
#                # restart
#                p.terminate()   
#                pr = multiprocessing.Process( name=p.name,
#                     target=canadian_flow_retrieval, args=(odir) ) 
#                proc_restarted.append( pr )
#                pr.start()
#                print( 'restarted : ', p.name )
#            else:
#                print( 'process: ', p.name, 'status file doesn\'t exist' )
#                p.terminate()   
#                pr = multiprocessing.Process( name=p.name,
#                     target=canadian_flow_retrieval, args=(odir) ) 
#                proc_restarted.append( pr )
#                pr.start()
#                print( 'process: ', p.name, 'restarted :' )
#        else:
#            print( 'process: ', p.name, ' is not alive.' )
#            p.terminate()   
#            pr = multiprocessing.Process( name=p.name,
#                 target=canadian_flow_retrieval, args=(odir) ) 
#            proc_restarted.append( pr )
#            print( 'process: ', p.name, 'restarted :' )
#            pr.start()
#
#    if proc_restarted:
#        for p in proc_restarted:
#            procs[ p.name ] = p

#for p in procs.values():
#    p.join()
#    print( '%s.exitcode = %s' % (p.name, p.exitcode) )

