import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2


class CameraViewer(Node):

    def __init__(self):
        super().__init__('camera_viewer')

        self.bridge = CvBridge()
        self.finger_number = 0

        # ✅ Subscribe to annotated image instead of raw camera
        self.subscription = self.create_subscription(
            Image,
            'gesture/image_annotated',
            self.image_callback,
            10
        )

        # Finger number subscriber
        self.gesture_sub = self.create_subscription(
            Int32,
            'gesture/number',
            self.gesture_callback,
            10
        )

        self.get_logger().info("Camera Viewer Started")

    # --------------------------------
    def gesture_callback(self, msg):
        self.finger_number = msg.data

    # --------------------------------
    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Draw finger count overlay
        cv2.putText(
            frame,
            f"Fingers: {self.finger_number}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera Viewer", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = CameraViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
