###############################################################################
#  Module name: TidesCurrentsProduct                                          #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:  12/19/2024                                          #
#                                                                             #
#  Last modification date:                                                    # 
#                                                                             #
#  Description: Abstract the Tides and Currents Products from NOAA's Center   #
#         for Operational Oceanographic Products and Services (CO-OPS).       #
#                                                                             #
###############################################################################

import os, logging
from string import *
from datetime import datetime, timedelta
import dateutil.parser
import pytz
from abc import ABC, ABCMeta, abstractmethod, abstractproperty

class TidesCurrentsProduct(ABC):
        """
           Abstract the Tides and Currents products
        """        
        __metaclass__ = ABCMeta

        @property
        def source(self):
            return self._source

        @source.setter
        def source(self, s):
            self._source = s

        @property
        def name(self):
            return self._name

        @name.setter
        def name(self, n):
            self._name = n

        @abstractmethod
        def __init__(self, filename ):
            pass
        
