###############################################################################
#  Module name: NOAAObservedWL                                                #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:  1/29/2025                                           #
#                                                                             #
#  Last modification date:                                                    # 
#                                                                             #
#  Description: Manages a NOAA observed water level json file                 #
#                                                                             #
###############################################################################

import glob, numpy as np, json

class NOAAObservedWL:
        """
           NOAA Observed water level json file
        """        

        def __init__(self, observedfile, datumfile ): 
           self._source = observedfile
           self._datumsource = datumfile

           with open(observedfile, 'r') as jfile: 
                self._data = json.load( jfile ) 
                if  'error' in self._data:
                    raise ValueError( self._data['error'][ 'message' ] + " : " + observedfile )

           with open(datumfile, 'r') as dfile: 
                self._datum = json.load( dfile ) 
                if  'error' in self._datum:
                    raise ValueError( self._datum['error'][ 'message' ] + " : " + datumfile )

        @property
        def source(self):
            return self._source

        @source.setter
        def source(self, s):
            self._source = s

        @property
        def data(self):
            return self._data

        @data.setter
        def data(self, d):
            self._data = d

        @property
        def datum(self):
            return self._datum

        @datum.setter
        def datum(self, d):
            self._datum = d
