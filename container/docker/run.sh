#!/bin/bash

: '
This script defines the behavior of the container. The gym-pybullet-drones 
repository is at /gym-pybullet-drones in the container filesystem. 

The python command is Python3.7

The command to run the Docker container should be as follows:
	docker run --rm -it -v"$(pwd)":/mnt connorfuhrman/gym-pybullet-drones:draft 
or similar tag/mounting point.

'

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
cd video* && sh ffmpeg_png2mp4.sh && mv video.mp4 /mnt
# =================================================================================