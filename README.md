```bash
# Prerequisite
Ubuntu 20.04
ROS2 foxy
Turtlebot3 packages (refer to turtlebot3 emanual)

# Terminal 1
git clone <your VAEplusDDPG fork repository>
cd ~/vaeplusddpg
colcon build --symlink-install

<If you have "dart" error during build, you should clone NAVIGATION/dart>
<If you have error during build, you just need to run "source opt/ros/foxy/setup.bash">

source install/setup.bash
cd src/turtlebot3_simulations/turtlebot3_gazebo
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage6.launch.py

# Terminal 2
ros2 run turtlebot3_drl environment

# Terminal 3
cd ~/vaeplusddpg/src/turtlebot3_drl/turtlebot3_drl/drl_agent
ros2 run turtlebot3_drl train_agent ddpg

<If you have pt files(pretrained model), you can run test as blow>
ros2 run turtlebot3_drl test 'ddpg_55_stage_6' 3600

# Terminal 4
ros2 run turtlebot3_drl gazebo_goals

<If you have error in Terminal 3, you just need to run that command again>



