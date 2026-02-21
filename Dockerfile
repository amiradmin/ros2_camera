FROM ros:jazzy-ros-base

# Install essential packages
RUN sed -i 's|http://.*archive.ubuntu.com|http://mirror.arvancloud.ir/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://mirror.arvancloud.ir/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --fix-missing \
    v4l-utils \
    libv4l-dev \
    python3-opencv \
    python3-pip \
    python3-venv \
    wget \
    sudo \
    ros-jazzy-cv-bridge \
    nano \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Download hand landmarker model
RUN wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
    -O /tmp/hand_landmarker.task

# Create a non-root user
RUN useradd -m -s /bin/bash amir && \
    echo "amir ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    usermod -a -G video,dialout amir

# Set up environment
ENV ROS_DOMAIN_ID=0
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Create workspace
WORKDIR /workspace
RUN mkdir -p /workspace/src && \
    chown -R amir:amir /workspace

# Copy requirements file
COPY --chown=amir:amir requirements.txt /tmp/requirements.txt

# Switch to non-root user
USER amir

# Create and activate virtual environment
RUN python3 -m venv /home/amir/venv
ENV PATH="/home/amir/venv/bin:$PATH"

# Install Python packages in virtual environment
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Setup ROS2 in bashrc with virtual environment
RUN echo '' >> /home/amir/.bashrc && \
    echo '# ROS 2 Workspace Setup' >> /home/amir/.bashrc && \
    echo 'source /opt/ros/jazzy/setup.bash' >> /home/amir/.bashrc && \
    echo 'source /home/amir/venv/bin/activate' >> /home/amir/.bashrc && \
    echo 'cd /workspace' >> /home/amir/.bashrc && \
    echo 'if [ -f "install/setup.bash" ]; then' >> /home/amir/.bashrc && \
    echo '    source install/setup.bash' >> /home/amir/.bashrc && \
    echo 'fi' >> /home/amir/.bashrc

WORKDIR /workspace

CMD ["/bin/bash"]