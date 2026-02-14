FROM yuanboshe/ros-humble-aicrobo:latest

# -----------------------------
# Install system dependencies
# -----------------------------
USER root

RUN apt-get update && apt-get install -y \
    v4l-utils \
    libv4l-dev \
    python3-opencv \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Allow camera access
RUN usermod -a -G video aicrobo

# -----------------------------
# Download MediaPipe Model
# -----------------------------
RUN wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
    -O /tmp/hand_landmarker.task

USER aicrobo

# -----------------------------
# Workspace Setup
# -----------------------------
WORKDIR /home/aicrobo/ros2_ws

# Install Python dependencies
COPY --chown=aicrobo:aicrobo requirements.txt /tmp/requirements.txt

RUN if [ -f /tmp/requirements.txt ]; then \
        pip install --no-cache-dir -r /tmp/requirements.txt; \
        rm /tmp/requirements.txt; \
    fi
RUN pip install mediapipe
RUN pip install --force-reinstall "numpy==1.24.3"
# -----------------------------
# ROS Environment Setup
# -----------------------------
RUN echo '' >> ~/.bashrc && \
    echo '# ROS 2 Workspace Setup' >> ~/.bashrc && \
    echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc && \
    echo 'cd /home/aicrobo/ros2_ws' >> ~/.bashrc && \
    echo 'if [ -f "install/setup.bash" ]; then' >> ~/.bashrc && \
    echo '    source install/setup.bash' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc
