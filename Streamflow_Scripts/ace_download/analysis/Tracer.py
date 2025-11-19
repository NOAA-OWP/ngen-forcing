###############################################################################
#  Module name: Tracer
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  7/12/2017                                         #
#                                                                             #
#  Description: Create a Python Tracer object for debugging purpose           #
#                                                                             #
###############################################################################

import sys, trace

def init():
	global theTracer
	theTracer = \
	     trace.Trace( ignoredirs=sys.path[1:], trace=True, count=False )
