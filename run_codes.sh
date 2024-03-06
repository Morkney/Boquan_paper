#!/bin/bash

#-------------------------------------------------
inputs=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30")
inputs=("5" "6" "9" "13" "17" "23" "24" "26" "28")
#-------------------------------------------------

for input in "${inputs[@]}"
do
  echo $input
  #python merger_tree_metals.py $input
  #python metal_alpha_plane.py $input
  #python merger_statistics.py $input
  #python merger_trajectories.py $input
  #python merger_tree_metals.py $input
  #python plume_analysis.py $input
  python SFH_with_radius2.py $input
done
