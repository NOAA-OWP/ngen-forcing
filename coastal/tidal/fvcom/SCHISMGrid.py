###############################################################################
#  Module name: SCHISMGrid                                                    #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:  12/24/2024                                          #
#                                                                             #
#  Last modification date:                                                    # 
#                                                                             #
#  Description: Manages a SCHISM hgrid file                                   #
#                                                                             #
###############################################################################

class SCHISMGrid:
        """
           SCHISM hgrid.gr3 file
        """        

        ZETA0 = {"leofs":173.5, "lmhofs" :175.0, "loofs":72.4, "lsofs":183.2 } 

        def __init__(self, hgridfile, z ): #z - to be added to the FVCOM zeta. Is this the depth?
           self.source = hgridfile
           self.zeta0 = z
           lon = []
           lat = []
           bnodes = []
           with open(hgridfile) as f:
               next(f)
               line = f.readline()
               ne = int(line.split()[0])
               nn = int(line.split()[1])
               for i in range(nn):
                  line = f.readline()
                  lon.append(float(line.split()[1]))
                  lat.append(float(line.split()[2]))
               for i in range(ne):
                  f.readline()
               line = f.readline()
               nbounds = int(line.split()[0])
               next(f)
               for i in range(nbounds):
                 line = f.readline()
                 nnodes = int(line.split()[0])
                 for j in range(nnodes):
                    bnodes.append(int(f.readline()))

           self.lons = [lon[b-1] for b in bnodes]
           self.lats = [lat[b-1] for b in bnodes]
           self.bnodes = bnodes

        @property
        def source(self):
            return self._source

        @source.setter
        def source(self, s):
            self._source = s

        @property
        def lats(self):
            return self._lats

        @lats.setter
        def lats(self, l):
            self._lats = l

        @property
        def lons(self):
            return self._lons

        @lons.setter
        def lons(self, l):
            self._lons = l

        @property
        def bnodes(self):
            return self._bnodes

        @bnodes.setter
        def bnodes(self, b):
            self._bnodes = b

        @property
        def zeta0(self):
            return self._zeta0

        @zeta0.setter
        def zeta0(self, z):
            self._zeta0 = z
