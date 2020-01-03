#!/bin/bash
set -e

# source environment
source "/opt/ros/melodic/setup.bash"
mkdir -p /catkin_ws/src
cd /catkin_ws
catkin_make
source "/catkin_ws/devel/setup.bash"
exec "$@"
