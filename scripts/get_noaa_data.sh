#!/bin/bash
echo $0

if [ -n "$1" ]
then
year=$1
else
year=`date +%Y`
fi

wget https://www1.ncdc.noaa.gov/pub/data/gsod/$year/gsod_$year.tar -O ../data/noaa/gsod_$year.tar