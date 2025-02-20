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

source install/setup.bash
cd src/turtlebot3_simulations/turtlebot3_gazebo
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage6.launch.py

# Terminal 2
cd
ros2 run turtlebot3_drl environment

# Terminal 3
cd ~/vaeplusddpg/src/turtlebot3_drl/turtlebot3_drl/drl_agent
ros2 run turtlebot3_drl train_agent ddpg

# Terminal 4
cd
ros2 run turtlebot3_drl gazebo_goals

If you have follow error, you should read "Issues" column.

File "/home/dmsgv1/vaeplusddpg/build/turtlebot3_drl/turtlebot3_drl/common/replaybuffer.py", line 16, in sample
    s_array = np.float32([array[0] for array in batch])
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (64,) + inhomogeneous part.


