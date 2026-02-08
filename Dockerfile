FROM yuanboshe/ros-humble-aicrobo:latest

WORKDIR /home/aicrobo/ros2_ws

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
