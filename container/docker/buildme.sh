#!/bin/bash


: '
Convenience script to tar the repo, build the docker container, and un-tar the
repo since the tarball is too large for GitHub. 

!!!Script must be run from the container/docker directory!!!
'

# Tar the repo above this script
tar -czvf gym-pybullet-drones.tar.gz \
	--exclude=../../../gym-pybullet-drones/.git/ \
	--exclude=../../../gym-pybullet-drones/ros2 \
	--exclude=../../../gym-pybullet-drones/container\
	../../../gym-pybullet-drones \
	
	
# Build the docker image
docker build -t connorfuhrman/gym-pybullet-drones:draft .

# Remove the tarball
rm gym-pybullet-drones.tar.gz