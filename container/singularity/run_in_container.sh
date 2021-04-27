#!/bin/sh

printf "I'm running inside the singularity container motha-fucka!\n"

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
