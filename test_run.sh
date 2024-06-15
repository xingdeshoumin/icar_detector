#!/bin/bash
colcon build
source install/setup.bash
ros2 run icar_detector detector
# colcon test
