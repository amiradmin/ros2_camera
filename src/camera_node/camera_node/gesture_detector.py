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

        # Subscribe camera
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        # Publish finger number
        self.number_pub = self.create_publisher(Int32, 'gesture/number', 10)

        # Publish annotated image
        self.image_pub = self.create_publisher(Image, 'gesture/image_annotated', 10)

        # Hand skeleton connections (MediaPipe topology)
        self.hand_connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        ]

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

    # ---------------------------------------------------
    def count_fingers(self, landmarks):

        tips = [4, 8, 12, 16, 20]
        count = 0

        if landmarks[4].x < landmarks[3].x:
            count += 1

        for tip in tips[1:]:
            if landmarks[tip].y < landmarks[tip - 2].y:
                count += 1

        return count

    # ---------------------------------------------------
    def draw_landmarks(self, frame, landmarks):

        h, w, _ = frame.shape

        points = []

        # Convert normalized coords → pixel coords
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

            # Draw joint
            cv2.circle(frame, (px, py), 4, (0,255,0), -1)

        # Draw skeleton
        for connection in self.hand_connections:
            pt1 = points[connection[0]]
            pt2 = points[connection[1]]
            cv2.line(frame, pt1, pt2, (255,0,0), 2)

    # ---------------------------------------------------
    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        finger_count = 0

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            finger_count = self.count_fingers(landmarks)

            # ⭐ Draw landmarks manually
            self.draw_landmarks(frame, landmarks)

        # Draw finger count text
        cv2.putText(
            frame,
            f"Fingers: {finger_count}",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # Publish finger count
        number_msg = Int32()
        number_msg.data = finger_count
        self.number_pub.publish(number_msg)

        # Publish annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.image_pub.publish(annotated_msg)


# ---------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    node = GestureDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
