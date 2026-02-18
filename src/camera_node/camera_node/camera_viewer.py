#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.bridge = CvBridge()

        # Create a named window with normal size
        cv2.namedWindow("Gesture Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Detection", 800, 600)

        # Subscribe to annotated images
        self.subscription = self.create_subscription(
            Image,
            'gesture/image_annotated',
            self.image_callback,
            10
        )

        self.get_logger().info("✅ Image Viewer Started - waiting for images...")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Show the frame
            cv2.imshow("Gesture Detection", frame)
            cv2.waitKey(1)  # 1ms delay, allows window to update

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def cleanup(self):
        cv2.destroyAllWindows()
        self.get_logger().info("Cleaned up windows")


def main(args=None):
    rclpy.init(args=args)
    node = ImageViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()