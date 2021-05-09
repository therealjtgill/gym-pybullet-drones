#!/bin/sh

printf "I'm running inside the singularity container motha-fucka!\n"

: '
# =================================================================================
# Example: Run fly.py for 5 seconds w/out GUI, make a video, and export that video
# to the current working directory where docker run is called. 

# Move to top-level of repo code
cd /gym-pybullet-drones 
# Execute the fly example
python3.7 examples/fly.py \
    --gui False \
    --record_video True \
    --duration_sec 5
# Convert captured PNGs to MP4 via ffmpeg
cd files/videos
# Assume that there is only one video* folder!!
cp ffmpeg_png2mp4.sh video*
cd video* && sh ffmpeg_png2mp4.sh
# =================================================================================
'

# Run learn.py --rllib to check PPO 
cd ~/gym-pybullet-drones-sandbox/gym-pybullet-drones
#python3.7 examples/learn.py --rllib=True 
python3.7 -m pip install -e .
python3.7 examples/learn_shoot_and_defend.py --num_workers 16
