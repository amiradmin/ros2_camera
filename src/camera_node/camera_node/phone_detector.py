#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Bool, String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from ultralytics import YOLO  # For phone detection


class PhoneDetector(Node):
    def __init__(self):
        super().__init__('phone_detector')

        # Declare parameters
        self.declare_parameter('model_path', '/tmp/yolov8n.pt')  # YOLO nano model
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('proximity_threshold', 100)  # pixels
        self.declare_parameter('show_gui', True)

        self.model_path = self.get_parameter('model_path').value
        self.confidence = self.get_parameter('confidence_threshold').value
        self.proximity_threshold = self.get_parameter('proximity_threshold').value
        self.show_gui = self.get_parameter('show_gui').value

        self.bridge = CvBridge()

        # Subscribe to camera images
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',  # or 'gesture/image_annotated' if you want landmarks
            self.image_callback,
            10
        )

        # Publishers
        self.phone_detected_pub = self.create_publisher(Bool, 'phone/detected', 10)
        self.phone_count_pub = self.create_publisher(Int32, 'phone/count', 10)
        self.phone_position_pub = self.create_publisher(Point, 'phone/position', 10)
        self.phone_hand_status_pub = self.create_publisher(String, 'phone/hand_status', 10)

        # Load YOLO model (for phone detection)
        self.load_model()

        # For hand tracking (simplified - uses color detection)
        self.hand_positions = []

        # Download YOLO model if not exists
        self.download_model()

        if self.show_gui:
            cv2.namedWindow("Phone Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Phone Detector", 800, 600)

        self.get_logger().info("✅ Phone Detector Started")

    def download_model(self):
        """Download YOLO model if not present"""
        if not os.path.exists(self.model_path):
            self.get_logger().info("Downloading YOLO model...")
            try:
                # You can also use a smaller model specifically for phones
                # For now, we'll use YOLOv8n
                import urllib.request
                url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
                urllib.request.urlretrieve(url, self.model_path)
                self.get_logger().info("✅ Model downloaded")
            except Exception as e:
                self.get_logger().error(f"Failed to download model: {e}")

    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info("✅ YOLO model loaded")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect_hands_simple(self, frame):
        """Simple hand detection using skin color"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask for skin
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Clean up mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hand_positions = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filter small areas
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                hand_positions.append({
                    'bbox': (x, y, w, h),
                    'center': center,
                    'area': cv2.contourArea(contour)
                })

                if self.show_gui:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)

        return hand_positions, frame

    def detect_phones(self, frame):
        """Detect phones using YOLO"""
        if self.model is None:
            return [], frame

        # Run YOLO inference
        results = self.model(frame, conf=self.confidence, classes=[67])  # class 67 is 'cell phone' in COCO

        phones = []
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    phone = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        'confidence': conf
                    }
                    phones.append(phone)

                    if self.show_gui:
                        # Draw phone bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"Phone {conf:.2f}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return phones, frame

    def check_phone_in_hand(self, phone, hands):
        """Check if a phone is being held by any hand"""
        phone_center = phone['center']

        for hand in hands:
            hand_center = hand['center']
            hand_bbox = hand['bbox']

            # Calculate distance between phone and hand
            distance = np.sqrt((phone_center[0] - hand_center[0]) ** 2 +
                               (phone_center[1] - hand_center[1]) ** 2)

            # Check if phone is within hand bounding box
            x, y, w, h = hand_bbox
            phone_in_hand_bbox = (x < phone_center[0] < x + w and
                                  y < phone_center[1] < y + h)

            if distance < self.proximity_threshold or phone_in_hand_bbox:
                return True, hand

        return False, None

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            original = frame.copy()

            # Detect hands
            hands, frame_with_hands = self.detect_hands_simple(frame)

            # Detect phones
            phones, frame_with_phones = self.detect_phones(frame_with_hands)

            # Analyze phone-hand interactions
            phones_in_hand = []
            phones_not_in_hand = []

            phone_hand_status = "No phones detected"

            if phones:
                for phone in phones:
                    in_hand, hand = self.check_phone_in_hand(phone, hands)

                    if in_hand:
                        phones_in_hand.append(phone)
                        # Draw hand-phone connection
                        cv2.line(frame_with_phones, phone['center'], hand['center'], (0, 255, 255), 2)
                        cv2.putText(frame_with_phones, "IN HAND",
                                    (phone['bbox'][0], phone['bbox'][1] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Publish phone position
                        pos_msg = Point()
                        pos_msg.x = float(phone['center'][0])
                        pos_msg.y = float(phone['center'][1])
                        pos_msg.z = 0.0
                        self.phone_position_pub.publish(pos_msg)
                    else:
                        phones_not_in_hand.append(phone)
                        cv2.putText(frame_with_phones, "DETECTED",
                                    (phone['bbox'][0], phone['bbox'][1] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Update status
                if phones_in_hand:
                    phone_hand_status = f"{len(phones_in_hand)} phone(s) in hand"
                    if phones_not_in_hand:
                        phone_hand_status += f", {len(phones_not_in_hand)} detected nearby"
                else:
                    phone_hand_status = f"{len(phones)} phone(s) detected but not in hand"

                # Publish detection status
                detected_msg = Bool()
                detected_msg.data = len(phones) > 0
                self.phone_detected_pub.publish(detected_msg)

                count_msg = Int32()
                count_msg.data = len(phones_in_hand)
                self.phone_count_pub.publish(count_msg)

                status_msg = String()
                status_msg.data = phone_hand_status
                self.phone_hand_status_pub.publish(status_msg)

            # Draw status on frame
            cv2.putText(frame_with_phones, f"Phones in hand: {len(phones_in_hand)}",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_with_phones, phone_hand_status,
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Show frame if GUI enabled
            if self.show_gui:
                cv2.imshow("Phone Detector", frame_with_phones)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def cleanup(self):
        if self.show_gui:
            cv2.destroyAllWindows()
        self.get_logger().info("Phone Detector cleaned up")


def main(args=None):
    rclpy.init(args=args)
    node = PhoneDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Phone Detector...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()