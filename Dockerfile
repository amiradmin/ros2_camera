FROM yuanboshe/ros-humble-aicrobo:latest

# Install camera dependencies as root
USER root
RUN apt-get update && apt-get install -y \
    v4l-utils \
    libv4l-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*
RUN usermod -a -G video aicrobo
USER aicrobo

# Set working directory
WORKDIR /home/aicrobo/ros2_ws

# Add ROS setup to bashrc
RUN echo '' >> ~/.bashrc && \
    echo '# ROS 2 Workspace Setup' >> ~/.bashrc && \
    echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc && \
    echo 'cd /home/aicrobo/ros2_ws' >> ~/.bashrc && \
    echo 'if [ -f "install/setup.bash" ]; then' >> ~/.bashrc && \
    echo '    source install/setup.bash' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc