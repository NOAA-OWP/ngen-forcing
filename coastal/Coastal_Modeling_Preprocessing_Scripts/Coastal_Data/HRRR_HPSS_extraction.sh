#!/bin/bash

path='hsi get /NCEPPROD/hpssprod/runhistory/rh'
pathway='/'
hrrr='hrrr.'
#If the time stamp is before 2022-06-17 18Z
#then the naming convention below is correct
hrrr_file_beg='com_hrrr_prod_hrrr.'
# Otherwise, the naming convention changes 
# with upgrades to the HRRR system itself
#hrrr_file_beg='com_hrrr_v4.1_hrrr.'


path='hsi get /NCEPPROD/hpssprod/runhistory/rh'
pathway='/'
hrrr='hrrr.'
hrrr_file_end='.wrf.tar'
tar_command='tar -xvf '


#Switch between CONUS and Alaska HRRR extraction
hrrr_file_mid='_alaska'
hrrr_file_mid='_conus'

mkdir tmp
cd tmp
pwd

# Change years, months, and day loops for HRRR extraction
for year in {2020..2020}
do
   for month in {03..03}
   do
      for day in {07..08}
         do echo 'Getting data for' $year$month$day '....'
         mkdir ../$hrrr$year$month$day$pathway
         $path$year$pathway$year$month$pathway$year$month$day$pathway$hrrr_file_beg$year$month$day$hrrr_file_mid'00-05'$hrrr_file_end
         $tar_command$hrrr_file_beg$year$month$day$hrrr_file_mid'00-05'$hrrr_file_end
         mv *wrfprsf*.ak.grib2 ../$hrrr$year$month$day$pathway
         rm -rf *
         $path$year$pathway$year$month$pathway$year$month$day$pathway$hrrr_file_beg$year$month$day$hrrr_file_mid'06-11'$hrrr_file_end
         $tar_command$hrrr_file_beg$year$month$day$hrrr_file_mid'06-11'$hrrr_file_end
         mv *wrfprsf*.ak.grib2 ../$hrrr$year$month$day$pathway
         rm -rf *
         $path$year$pathway$year$month$pathway$year$month$day$pathway$hrrr_file_beg$year$month$day$hrrr_file_mid'12-17'$hrrr_file_end
         $tar_command$hrrr_file_beg$year$month$day$hrrr_file_mid'12-17'$hrrr_file_end
         mv *wrfprsf*.ak.grib2 ../$hrrr$year$month$day$pathway
         rm -rf *
         $path$year$pathway$year$month$pathway$year$month$day$pathway$hrrr_file_beg$year$month$day$hrrr_file_mid'18-23'$hrrr_file_end
         $tar_command$hrrr_file_beg$year$month$day$hrrr_file_mid'18-23'$hrrr_file_end
         mv *wrfprsf*.ak.grib2 ../$hrrr$year$month$day$pathway
         rm -rf *
      done
   done
done
printf "\n"
