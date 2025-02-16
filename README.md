```bash
# Prerequisite
Ubuntu 20.04
ROS2 foxy
Turtlebot3 packages (refer to turtlebot3 emanual)

# Terminal 1
git clone git@github.com:uiseoklee/VAEplusDDPG.git
cd ~/VAEplusDDPG
colcon build --symlink-install
source install/setup.bash
cd src/turtlebot3_simulations/turtlebot3_gazebo
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage6.launch.py

# Terminal 2
cd
ros2 run turtlebot3_drl environment

# Terminal 3
cd ~/VAEplusDDPG/src/turtlebot3_drl/turtlebot3_drl/drl_agent
ros2 run turtlebot3_drl train_agent ddpg

# Terminal 4
cd
ros2 run turtlebot3_drl gazebo_goals
