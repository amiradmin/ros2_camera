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
from ultralytics import YOLO


class PhoneCrossingDetector(Node):
    def __init__(self):
        super().__init__('phone_crossing_detector')

        # Declare parameters
        self.declare_parameter('line_position', 0.5)  # 0.5 = middle of frame
        self.declare_parameter('threshold', 30)  # pixels for crossing detection
        self.declare_parameter('show_gui', True)
        self.declare_parameter('model_path', '/tmp/yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('proximity_threshold', 100)  # pixels for phone-hand proximity

        self.line_position = self.get_parameter('line_position').value
        self.threshold = self.get_parameter('threshold').value
        self.show_gui = self.get_parameter('show_gui').value
        self.model_path = self.get_parameter('model_path').value
        self.confidence = self.get_parameter('confidence_threshold').value
        self.proximity_threshold = self.get_parameter('proximity_threshold').value

        self.bridge = CvBridge()

        # Subscribe to camera images
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers - Phone detection
        self.phone_detected_pub = self.create_publisher(Bool, 'phone/detected', 10)
        self.phone_count_pub = self.create_publisher(Int32, 'phone/count', 10)
        self.phone_position_pub = self.create_publisher(Point, 'phone/position', 10)
        self.phone_hand_status_pub = self.create_publisher(String, 'phone/hand_status', 10)

        # Publishers - Crossing detection
        self.crossing_pub = self.create_publisher(Int32, 'gesture/crossing_count', 10)
        self.phone_crossing_pub = self.create_publisher(Bool, 'phone/crossing_detected', 10)

        # Load YOLO model
        self.load_model()

        # Crossing detection variables
        self.crossing_count = 0
        self.hand_positions = {}  # Track hand positions by ID
        self.last_side = {}  # Track last side for each hand
        self.next_hand_id = 0
        self.previous_centers = {}
        self.tracking_distance = 50  # Max distance to consider same hand

        # Phone tracking
        self.phones_in_hand = []
        self.phone_crossing_count = 0
        self.tracked_phones = {}  # Track phones by ID

        if self.show_gui:
            cv2.namedWindow("Phone & Crossing Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Phone & Crossing Detector", 800, 600)

        self.get_logger().info("✅ Phone & Crossing Detector Started")
        self.get_logger().info(f"   Line at {self.line_position * 100:.0f}%")

    def load_model(self):
        """Load YOLO model"""
        try:
            if not os.path.exists(self.model_path):
                self.get_logger().info("Downloading YOLO model...")
                # YOLO will download automatically if not present
            self.model = YOLO(self.model_path)
            self.get_logger().info("✅ YOLO model loaded")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect_phones(self, frame):
        """Detect phones using YOLO"""
        if self.model is None:
            return [], frame

        # Run YOLO inference (class 67 is 'cell phone' in COCO)
        results = self.model(frame, conf=self.confidence, classes=[67])

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
                        'confidence': conf,
                        'id': None  # Will be assigned if tracked
                    }
                    phones.append(phone)

                    if self.show_gui:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"Phone {conf:.2f}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return phones, frame

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

    def get_hand_center_from_contours(self, frame):
        """Alternative hand detection using green dots (from gesture_detector)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define green color range (for the landmarks from gesture_detector)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Create mask for green dots
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours of green dots
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hands = []
        if len(contours) > 5:  # Enough dots to form a hand
            all_points = []
            for contour in contours:
                if cv2.contourArea(contour) > 5:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        all_points.append((cx, cy))

            if len(all_points) > 5:
                # Calculate average center
                center_x = sum(p[0] for p in all_points) / len(all_points)
                center_y = sum(p[1] for p in all_points) / len(all_points)

                hands.append({
                    'center': (int(center_x), int(center_y)),
                    'bbox': (int(center_x - 50), int(center_y - 50), 100, 100),
                    'from_landmarks': True
                })

        return hands

    def find_matching_hand(self, current_center, frame_shape):
        """Find if current hand matches any previously tracked hand"""
        h, w = frame_shape[:2]
        current_pixel = (int(current_center[0]), int(current_center[1]))

        min_dist = float('inf')
        matching_id = None

        for hand_id, prev_center in self.previous_centers.items():
            prev_pixel = (int(prev_center[0]), int(prev_center[1]))
            dist = np.sqrt((current_pixel[0] - prev_pixel[0]) ** 2 +
                           (current_pixel[1] - prev_pixel[1]) ** 2)

            if dist < self.tracking_distance and dist < min_dist:
                min_dist = dist
                matching_id = hand_id

        return matching_id

    def get_side(self, center_x, frame_width):
        """Determine which side of the line the hand/phone is on"""
        line_x = frame_width * self.line_position
        if center_x < line_x:
            return "LEFT"
        else:
            return "RIGHT"

    def check_crossing(self, object_id, current_side, center_x, frame_width, object_type="hand"):
        """Check if object crossed the line"""
        if object_id in self.last_side:
            last_side = self.last_side[object_id]
            line_x = frame_width * self.line_position

            # Check if crossed (changed side)
            if last_side != current_side:
                # Calculate crossing threshold in pixels
                dist_to_line = abs(center_x - line_x)

                if dist_to_line < self.threshold:
                    if object_type == "hand":
                        self.crossing_count += 1
                        self.get_logger().info(f"✅ Hand {object_id} crossed! Total: {self.crossing_count}")

                        # Publish crossing count
                        msg = Int32()
                        msg.data = self.crossing_count
                        self.crossing_pub.publish(msg)
                    else:
                        self.phone_crossing_count += 1
                        self.get_logger().info(
                            f"📱 Phone {object_id} crossed! Phone crossings: {self.phone_crossing_count}")

                        # Publish phone crossing
                        phone_cross_msg = Bool()
                        phone_cross_msg.data = True
                        self.phone_crossing_pub.publish(phone_cross_msg)

                    return True
        return False

    def draw_crossing_line(self, frame):
        """Draw the crossing line on the frame"""
        h, w = frame.shape[:2]
        line_x = int(w * self.line_position)

        # Draw vertical line
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 3)
        cv2.putText(frame, "CROSSING LINE", (line_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return line_x

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            h, w = frame.shape[:2]

            # Draw crossing line
            line_x = self.draw_crossing_line(frame)

            # Method 1: Detect hands using skin color
            skin_hands, frame = self.detect_hands_simple(frame)

            # Method 2: Detect hands from green landmarks (if available)
            landmark_hands = self.get_hand_center_from_contours(frame)

            # Combine both hand detection methods
            all_hands = skin_hands + landmark_hands

            # Detect phones
            phones, frame = self.detect_phones(frame)

            # Track hands and check for crossings
            current_hand_centers = {}

            for i, hand in enumerate(all_hands):
                hand_center = hand['center']

                # Find matching hand
                hand_id = self.find_matching_hand(hand_center, frame.shape)

                if hand_id is None:
                    hand_id = self.next_hand_id
                    self.next_hand_id += 1

                current_hand_centers[hand_id] = hand_center

                # Get current side
                current_side = self.get_side(hand_center[0], w)

                # Draw hand info
                cv2.circle(frame, hand_center, 8, (255, 0, 255), -1)
                cv2.putText(frame, f"Hand {hand_id}",
                            (hand_center[0] + 10, hand_center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Check for crossing
                self.check_crossing(hand_id, current_side, hand_center[0], w, "hand")

                # Update last side
                self.last_side[hand_id] = current_side

            # Update previous centers
            self.previous_centers = current_hand_centers

            # Process phones and check if in hand
            phones_in_hand = []
            phone_hand_status = "No phones detected"

            if phones:
                for phone in phones:
                    in_hand, hand = self.check_phone_in_hand(phone, all_hands)

                    if in_hand:
                        phones_in_hand.append(phone)
                        # Draw hand-phone connection
                        cv2.line(frame, phone['center'], hand['center'], (0, 255, 255), 2)
                        cv2.putText(frame, "IN HAND",
                                    (phone['bbox'][0], phone['bbox'][1] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Check if phone crosses line while in hand
                        phone_side = self.get_side(phone['center'][0], w)
                        phone_id = hash(tuple(phone['center'])) % 1000  # Simple ID
                        self.check_crossing(phone_id, phone_side, phone['center'][0], w, "phone")
                    else:
                        cv2.putText(frame, "DETECTED",
                                    (phone['bbox'][0], phone['bbox'][1] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Publish phone position
                    pos_msg = Point()
                    pos_msg.x = float(phone['center'][0])
                    pos_msg.y = float(phone['center'][1])
                    pos_msg.z = 0.0
                    self.phone_position_pub.publish(pos_msg)

                # Update status
                if phones_in_hand:
                    phone_hand_status = f"{len(phones_in_hand)} phone(s) in hand"
                else:
                    phone_hand_status = f"{len(phones)} phone(s) detected"

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

            # Draw all counters on frame
            cv2.putText(frame, f"Hand Crossings: {self.crossing_count}",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Phone Crossings: {self.phone_crossing_count}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, phone_hand_status,
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show frame if GUI enabled
            if self.show_gui:
                cv2.imshow("Phone & Crossing Detector", frame)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def cleanup(self):
        if self.show_gui:
            cv2.destroyAllWindows()
        self.get_logger().info("Phone & Crossing Detector cleaned up")


def main(args=None):
    rclpy.init(args=args)
    node = PhoneCrossingDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Phone & Crossing Detector...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()