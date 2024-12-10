#!/bin/bash

# List of elements
elements=("bonsai" "room" "counter" "kitchen" "garden" "stump" "bicycle")
frame_nums=(24 10 1 1 3 5 1)
#x=(800 220 450 290 260 190 380)
#y=(170 260 17 110 50 40 180)
x=(440 450 680 450 350 330 370)
y=(190 300 175 250 140 50 200)

# Path to the validation_video.py script
script_path="python scripts/validation_video.py"

# Path to the runs directory
#runs_path="runs/runs_iNGP_360_B=2to20/"
runs_path="runs/runs_iNGP_360_B=2to20/"

# Path to the videos directory
videos_path="videos/"

# Loop through the lists simultaneously
for ((i=0; i<${#elements[@]}; i++)); do
    element="${elements[i]}"
    element_runs_path1="${runs_path}trial_ngp_ANR_360_${element}/"
    element_runs_path2="${runs_path}trial_ngp_ANR_OHPM_360_${element}/"
    video_file="${videos_path}${element}.mp4"
    frame_num="${frame_nums[i]}"
    x_val="${x[i]}"
    y_val="${y[i]}"
    
    # Run the validation_video.py script for the current element
    $script_path "${element_runs_path1}" "${element_runs_path2}" "${video_file}" \
     --frame_num "$frame_num" --x "$x_val" --y "$y_val"
done