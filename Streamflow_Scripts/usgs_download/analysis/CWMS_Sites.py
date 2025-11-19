###############################################################################
#  Module name: CWMS_Sites                                                    #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  09/05/2019                                        #
#                                                                             #
#  Description: manage the ACE CWMS sites in CSV format                       #
#                                                                             #
#  Updated by: Donald W Johnson (donald.w.johnson@noaa.gov)                   #
#                                                                             #
#  Update Description: Change column names to match new site file             #
#                                                                             #
###############################################################################

import csv

class CWMS_Sites:
        """
           Store CWMS site information 
        """        
        def __init__(self, csvSitefile ):
           """
              Initialize the CWMS_Sites object with a given
              filename
           """
           self.source = csvSitefile

           self._office_name1_to_index = dict()
           with open( csvSitefile, mode='r') as csvsite_file: 
                   csvsite_reader = csv.DictReader( csvsite_file )
                   line_count = 0
                   for row in csvsite_reader:
                           if line_count == 0:
                              print('Column names are ' + ", ".join(row))
                              line_count += 1
#                           print('\t' + row["office"] + " " + row["name_1"] )
                           if row["Office"] in self._office_name1_to_index:
                              self._office_name1_to_index[                  \
                                          row["Office"] ][row["gage"] ] = \
                                        row[ "NIDID" ]
                           else:
                              self._office_name1_to_index[ row["Office"] ] = \
                                 dict( { row["gage"] : \
                                        row[ "NIDID" ] } )

                           line_count += 1

                   print('Processed ' + str( line_count ) + ' lines.')
                   self.office_name1_to_index = self._office_name1_to_index


        @property
        def source(self):
            return self._source

        @source.setter
        def source(self, s):
            self._source = s


        @property
        def office_name1_to_index(self):
            return self._office_name1_to_index

        @office_name1_to_index.setter
        def office_name1_to_index(self, o):
            self._office_name1_to_index=o

        def getIndex(self, office, name1 ):
            return self.office_name1_to_index[ office][name1 ]
