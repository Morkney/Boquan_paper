#!/bin/bash

#-------------------------------------------------
#inputs=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30")
inputs=("24")
#-------------------------------------------------

for input in "${inputs[@]}"
do
  echo $input
  for snapshot in {127..40}
  do
    echo ">    $snapshot"
    python gas_homogeneity.py $input $snapshot
  python gas_movie.py $input
  done
done
