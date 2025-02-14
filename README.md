git clone git@github.com:uiseoklee/VAEplusDDPG.git
cd VAEplusDDPG
colcon build --symlink-install
source install/setup.bash

# Terminal 1
cd src/turtlebot3_simulations/turtlebot3_gazebo/
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage6.launch.py

# Terminal 2
ros2 run turtlebot3_drl environment

# Terminal 3
ros2 run turtlebot3_drl train_agent ddpg

# Terminal 4
ros2 run turtlebot3_drl gazebo_goals
