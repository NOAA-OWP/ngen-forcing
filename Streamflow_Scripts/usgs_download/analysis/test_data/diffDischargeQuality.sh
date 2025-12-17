#!/usr/bin/env bash

module load nco
module load cdo
fullO=$1
splitO=$2
rm -f diff.txt $fullO.txt $splitO.txt

#ncdump -b f $fullO > $fullO.txt
#ncdump -b f $splitO > $splitO.txt

ncdump -v discharge,discharge_quality -p 4 $fullO > $fullO.txt
ncdump -v discharge,discharge_quality -p 4 $splitO > $splitO.txt

diff $fullO.txt $splitO.txt > diff.txt
filesize=$(stat -c%s diff.txt)
if [[ $filesize -ne 0 ]]; then
  echo "Diff found"
  more diff.txt
else
  echo "Diff not found."
fi
