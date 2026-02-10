from setuptools import setup
import os

package_name = 'camera_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         [os.path.join('resource', package_name)]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'opencv-python', 'cv_bridge', 'mediapipe', 'rclpy'],
    zip_safe=True,
    maintainer='aicrobo',
    maintainer_email='aicrobo@todo.todo',
    description='Camera nodes for ROS2',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'camera_publisher = camera_node.camera_publisher:main',
            'camera_viewer = camera_node.camera_viewer:main',
            'gesture_detector = camera_node.gesture_detector:main',
        ],
    },
)
