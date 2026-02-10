import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GestureDetector(Node):

    def __init__(self):
        super().__init__('gesture_detector')

        self.bridge = CvBridge()

        # ROS subscriber
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        # ROS publisher
        self.number_pub = self.create_publisher(Int32, 'gesture/number', 10)

        # Load MediaPipe model
        base_options = python.BaseOptions(
            model_asset_path='/tmp/hand_landmarker.task'
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        self.get_logger().info("Gesture Detector Started")

    # --------------------------------
    def count_fingers(self, landmarks):
        tips = [4, 8, 12, 16, 20]
        count = 0

        # Thumb
        if landmarks[4].x < landmarks[3].x:
            count += 1

        # Other fingers
        for tip in tips[1:]:
            if landmarks[tip].y < landmarks[tip - 2].y:
                count += 1

        return count

    # --------------------------------
    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)

        finger_count = 0

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            finger_count = self.count_fingers(landmarks)

        number_msg = Int32()
        number_msg.data = finger_count
        self.number_pub.publish(number_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GestureDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
