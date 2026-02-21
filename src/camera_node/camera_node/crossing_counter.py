#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np


class CrossingCounter(Node):
    def __init__(self):
        super().__init__('crossing_counter')

        # Declare parameters
        self.declare_parameter('line_position', 0.5)  # 0.5 = middle of frame (50%)
        self.declare_parameter('threshold', 30)  # pixels threshold for crossing detection
        self.declare_parameter('show_gui', True)

        self.line_position = self.get_parameter('line_position').value
        self.threshold = self.get_parameter('threshold').value
        self.show_gui = self.get_parameter('show_gui').value

        self.bridge = CvBridge()

        # Subscribe to annotated images from gesture detector
        self.subscription = self.create_subscription(
            Image,
            'gesture/image_annotated',
            self.image_callback,
            10
        )

        # Publisher for crossing count
        self.crossing_pub = self.create_publisher(Int32, 'gesture/crossing_count', 10)

        # Crossing detection variables
        self.crossing_count = 0
        self.hand_positions = {}  # Track hand positions by ID
        self.last_side = {}  # Track last side for each hand
        self.next_hand_id = 0

        # For tracking movement
        self.previous_centers = {}
        self.tracking_distance = 50  # Max distance to consider same hand

        if self.show_gui:
            cv2.namedWindow("Crossing Counter", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Crossing Counter", 800, 600)

        self.get_logger().info(f"✅ Crossing Counter Started - line at {self.line_position * 100:.0f}%")
        self.get_logger().info(f"   Threshold: {self.threshold}px")

    def get_hand_center(self, landmarks):
        """Calculate center point of hand from landmarks"""
        if not landmarks:
            return None

        # Average all landmark points to get center
        x_sum = sum([lm.x for lm in landmarks])
        y_sum = sum([lm.y for lm in landmarks])
        center_x = x_sum / len(landmarks)
        center_y = y_sum / len(landmarks)

        return (center_x, center_y)

    def get_side(self, center_x, frame_width):
        """Determine which side of the line the hand is on"""
        line_x = frame_width * self.line_position
        if center_x < line_x:
            return "LEFT"
        else:
            return "RIGHT"

    def find_matching_hand(self, current_center, frame_shape):
        """Find if current hand matches any previously tracked hand"""
        h, w = frame_shape[:2]
        current_pixel = (int(current_center[0] * w), int(current_center[1] * h))

        min_dist = float('inf')
        matching_id = None

        for hand_id, prev_center in self.previous_centers.items():
            prev_pixel = (int(prev_center[0] * w), int(prev_center[1] * h))
            dist = np.sqrt((current_pixel[0] - prev_pixel[0]) ** 2 +
                           (current_pixel[1] - prev_pixel[1]) ** 2)

            if dist < self.tracking_distance and dist < min_dist:
                min_dist = dist
                matching_id = hand_id

        return matching_id

    def draw_crossing_line(self, frame):
        """Draw the crossing line on the frame"""
        h, w = frame.shape[:2]
        line_x = int(w * self.line_position)

        # Draw vertical line
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 3)

        # Draw line label
        cv2.putText(
            frame,
            "CROSSING LINE",
            (line_x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        return line_x

    def draw_crossing_count(self, frame):
        """Draw the crossing count on the frame"""
        cv2.putText(
            frame,
            f"Crossings: {self.crossing_count}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

    def check_crossing(self, hand_id, current_side, center_x, frame_width):
        """Check if hand crossed the line"""
        if hand_id in self.last_side:
            last_side = self.last_side[hand_id]
            line_x = frame_width * self.line_position

            # Check if crossed (changed side)
            if last_side != current_side:
                # Calculate crossing threshold in pixels
                pixel_x = center_x * frame_width
                dist_to_line = abs(pixel_x - line_x)

                if dist_to_line < self.threshold:
                    self.crossing_count += 1
                    self.get_logger().info(f"✅ Hand {hand_id} crossed! Total: {self.crossing_count}")

                    # Publish crossing count
                    msg = Int32()
                    msg.data = self.crossing_count
                    self.crossing_pub.publish(msg)

                    return True
        return False

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            h, w = frame.shape[:2]

            # Draw crossing line
            line_x = self.draw_crossing_line(frame)

            # For hand detection, we need to detect hands
            # Since we're subscribing to annotated images, we can use color-based detection
            # or we can modify gesture_detector to publish hand positions

            # Simplified: Use color detection to find hands (green dots from gesture_detector)
            # Find green circles (the landmarks drawn by gesture_detector)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define green color range (for the landmarks)
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([80, 255, 255])

            # Create mask for green dots
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Find contours of green dots
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 5:  # Enough dots to form a hand
                # Calculate center of all green dots (hand position)
                all_points = []
                for contour in contours:
                    if cv2.contourArea(contour) > 5:  # Filter small noise
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            all_points.append((cx, cy))

                if len(all_points) > 5:
                    # Calculate average center
                    center_x = sum(p[0] for p in all_points) / len(all_points)
                    center_y = sum(p[1] for p in all_points) / len(all_points)

                    # Convert to normalized coordinates
                    norm_center = (center_x / w, center_y / h)

                    # Find matching hand
                    hand_id = self.find_matching_hand(norm_center, frame.shape)

                    if hand_id is None:
                        hand_id = self.next_hand_id
                        self.next_hand_id += 1
                        self.get_logger().info(f"🆕 New hand detected: {hand_id}")

                    # Update tracking
                    self.previous_centers[hand_id] = norm_center

                    # Get current side
                    current_side = self.get_side(center_x, w)

                    # Draw hand center
                    cv2.circle(frame, (int(center_x), int(center_y)), 8, (255, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"Hand {hand_id}",
                        (int(center_x) + 10, int(center_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2
                    )

                    # Check for crossing
                    self.check_crossing(hand_id, current_side, center_x / w, w)

                    # Update last side
                    self.last_side[hand_id] = current_side

            # Draw crossing count
            self.draw_crossing_count(frame)

            # Show frame if GUI enabled
            if self.show_gui:
                cv2.imshow("Crossing Counter", frame)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def cleanup(self):
        if self.show_gui:
            cv2.destroyAllWindows()
        self.get_logger().info("Crossing Counter cleaned up")


def main(args=None):
    rclpy.init(args=args)
    node = CrossingCounter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Crossing Counter...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()