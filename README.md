# ROS2 Camera & Gesture System (Docker Setup)

This project runs ROS2 camera, gesture detection, and viewer nodes inside a Docker container.

---

## 📦 Prerequisites

Make sure the following are installed on the host system:

- Docker
- Docker Compose
- ROS2 workspace inside the container
- X11 support for GUI applications

---

## 🖥️ Enable X11 Display (Host Machine)

Allow Docker containers to access the host display:

```bash
xhost +local:docker



# Grant X11 permissions (run on host)
xhost +local:docker


# Build the Docker image
docker-compose build

# Start the container
docker-compose up -d


docker exec -it ros2_web_dev bash


cd /workspace
colcon build --symlink-install
source install/setup.bash



docker exec -it ros2_web_dev bash
cd /workspace && source install/setup.bash && ros2 run camera_node camera_publisher




docker exec -it ros2_web_dev bash 
cd /workspace && source install/setup.bash && ros2 run camera_node gesture_detector


docker exec -it ros2_web_dev bash
cd /workspace && source install/setup.bash && ros2 run camera_node camera_viewer


ros2 topic echo gesture/number







docker exec -it ros2_web_dev bash
cd /workspace && source install/setup.bash && ros2 run camera_node camera_publisher

docker exec -it ros2_web_dev bash 
cd /workspace && source install/setup.bash && ros2 run camera_node gesture_detector


docker exec -it ros2_web_dev bash
cd /workspace && source install/setup.bash && ros2 run camera_node crossing_counter


docker exec -it ros2_web_dev bash
cd /workspace && source install/setup.bash && ros2 topic echo /gesture/crossing_count



