#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Float32, Bool, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class CrossingCounter(Node):
    def __init__(self):
        super().__init__('crossing_counter')

        # Declare parameters
        self.declare_parameter('line_position', 0.5)  # 0.5 = middle of frame (50%)
        self.declare_parameter('threshold', 30)  # pixels threshold for crossing detection
        self.declare_parameter('show_gui', True)
        self.declare_parameter('pixels_per_cm', 10.0)  # Calibration: pixels per centimeter
        self.declare_parameter('reference_object_width_cm', 20.0)  # Width of reference object in cm
        self.declare_parameter('enable_calibration_mode', False)  # Set True to calibrate
        self.declare_parameter('alert_cooldown', 2.0)  # Seconds between alerts
        self.declare_parameter('use_horizontal_line', True)  # Use horizontal line instead of vertical
        self.declare_parameter('display_width', 1280)  # Target display width
        self.declare_parameter('display_height', 720)  # Target display height
        self.declare_parameter('screen_width', 1920)  # Screen width for max window size
        self.declare_parameter('screen_height', 1080)  # Screen height for max window size

        self.line_position = self.get_parameter('line_position').value
        self.threshold = self.get_parameter('threshold').value
        self.show_gui = self.get_parameter('show_gui').value
        self.pixels_per_cm = self.get_parameter('pixels_per_cm').value
        self.reference_width_cm = self.get_parameter('reference_object_width_cm').value
        self.calibration_mode = self.get_parameter('enable_calibration_mode').value
        self.alert_cooldown = self.get_parameter('alert_cooldown').value
        self.use_horizontal = self.get_parameter('use_horizontal_line').value
        self.display_width = self.get_parameter('display_width').value
        self.display_height = self.get_parameter('display_height').value
        self.screen_width = self.get_parameter('screen_width').value
        self.screen_height = self.get_parameter('screen_height').value

        self.bridge = CvBridge()

        # Subscribe to annotated images from gesture detector
        self.subscription = self.create_subscription(
            Image,
            'gesture/image_annotated',
            self.image_callback,
            10
        )

        # Subscribe to crossing count from phone detector to sync
        self.crossing_sub = self.create_subscription(
            Int32,
            'gesture/crossing_count',
            self.crossing_count_callback,
            10
        )

        # Publishers
        self.crossing_pub = self.create_publisher(Int32, 'gesture/crossing_count', 10)
        self.distance_pub = self.create_publisher(Float32, 'gesture/distance_to_line', 10)
        self.alert_pub = self.create_publisher(String, 'gesture/crossing_alert', 10)
        self.crossing_side_pub = self.create_publisher(String, 'gesture/crossing_side', 10)

        # Crossing detection variables
        self.crossing_count = 0
        self.hand_positions = {}  # Track hand positions by ID
        self.last_side = {}  # Track last side for each hand
        self.next_hand_id = 0

        # For tracking movement
        self.previous_centers = {}
        self.tracking_distance = 50  # Max distance to consider same hand

        # Alert system
        self.last_alert_time = 0
        self.alert_active = False
        self.alert_message = ""
        self.alert_duration = 1.0  # Show alert for 1 second

        # Calibration variables
        self.calibration_points = []
        self.calibration_complete = False

        # Calculate max window size (90% of screen)
        self.max_width = int(self.screen_width * 0.9)
        self.max_height = int(self.screen_height * 0.9)

        # Ensure display size doesn't exceed screen
        self.display_width = min(self.display_width, self.max_width)
        self.display_height = min(self.display_height, self.max_height)

        if self.show_gui:
            cv2.namedWindow("Crossing Counter", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Crossing Counter", self.display_width, self.display_height)

            if self.calibration_mode:
                cv2.setMouseCallback("Crossing Counter", self.mouse_callback)
                self.get_logger().info("📏 CALIBRATION MODE: Click and drag to select a reference object")

        line_type = "HORIZONTAL" if self.use_horizontal else "VERTICAL"
        self.get_logger().info(f"✅ Crossing Counter Started - {line_type} line at {self.line_position * 100:.0f}%")
        self.get_logger().info(f"   Threshold: {self.threshold}px ({self.px_to_cm(self.threshold):.1f} cm)")
        self.get_logger().info(f"   Calibration: {self.pixels_per_cm:.1f} pixels/cm")
        self.get_logger().info(f"   Display: {self.display_width}x{self.display_height}")

    def px_to_cm(self, pixels):
        """Convert pixels to centimeters using calibration"""
        if self.pixels_per_cm > 0:
            return pixels / self.pixels_per_cm
        return 0

    def cm_to_px(self, cm):
        """Convert centimeters to pixels using calibration"""
        return cm * self.pixels_per_cm

    def crossing_count_callback(self, msg):
        """Sync crossing count with other nodes"""
        if msg.data != self.crossing_count:
            self.crossing_count = msg.data
            self.get_logger().info(f"🔄 Synced crossing count: {self.crossing_count}")

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for calibration mode"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.calibration_points = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            if len(self.calibration_points) == 1:
                self.calibration_points.append((x, y))
                self.calibrate_from_points()

    def calibrate_from_points(self):
        """Calibrate pixels_per_cm using selected points"""
        if len(self.calibration_points) == 2:
            x1, y1 = self.calibration_points[0]
            x2, y2 = self.calibration_points[1]

            # Calculate pixel distance
            pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if pixel_distance > 0:
                self.pixels_per_cm = pixel_distance / self.reference_width_cm
                self.calibration_complete = True
                self.get_logger().info(f"✅ Calibration complete: {self.pixels_per_cm:.2f} pixels/cm")
                self.get_logger().info(f"   {pixel_distance:.1f} pixels = {self.reference_width_cm} cm")

    def get_side(self, center, frame_size):
        """Determine which side of the line the hand is on"""
        if self.use_horizontal:
            # For horizontal line: ABOVE or BELOW
            line_y = frame_size[0] * self.line_position
            if center[1] < line_y:
                return "ABOVE"
            else:
                return "BELOW"
        else:
            # For vertical line: LEFT or RIGHT
            line_x = frame_size[1] * self.line_position
            if center[0] < line_x:
                return "LEFT"
            else:
                return "RIGHT"

    def find_matching_hand(self, current_center, frame_shape):
        """Find if current hand matches any previously tracked hand"""
        min_dist = float('inf')
        matching_id = None

        for hand_id, prev_center in self.previous_centers.items():
            dist = np.sqrt((current_center[0] - prev_center[0]) ** 2 +
                           (current_center[1] - prev_center[1]) ** 2)

            if dist < self.tracking_distance and dist < min_dist:
                min_dist = dist
                matching_id = hand_id

        return matching_id

    def draw_crossing_line(self, frame):
        """Draw the crossing line on the frame"""
        h, w = frame.shape[:2]

        if self.use_horizontal:
            # Horizontal line
            line_pos = int(h * self.line_position)

            # Draw horizontal line
            cv2.line(frame, (0, line_pos), (w, line_pos), (0, 255, 255), 3)

            # Draw line label
            cv2.putText(
                frame,
                "CROSSING LINE",
                (30, line_pos - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # Draw threshold zone
            threshold_px = self.threshold
            cv2.rectangle(frame,
                          (0, line_pos - threshold_px),
                          (w, line_pos + threshold_px),
                          (0, 255, 255), 1, cv2.LINE_AA)

            # Add threshold label
            # cv2.putText(
            #     frame,
            #     f"±{self.threshold}px ({self.px_to_cm(self.threshold):.1f}cm)",
            #     (30, line_pos + 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 255),
            #     1
            # )

            return line_pos
        else:
            # Vertical line
            line_pos = int(w * self.line_position)
            cv2.line(frame, (line_pos, 0), (line_pos, h), (0, 255, 255), 3)
            cv2.putText(frame, "CROSSING LINE", (line_pos + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            threshold_px = self.threshold
            cv2.rectangle(frame,
                          (line_pos - threshold_px, 0),
                          (line_pos + threshold_px, h),
                          (0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, f"±{self.threshold}px ({self.px_to_cm(self.threshold):.1f}cm)",
                        (line_pos - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            return line_pos

    def draw_distance_info(self, frame, hand_center, line_pos):
        """Draw distance from hand to crossing line in cm"""
        if self.use_horizontal:
            # Distance to horizontal line
            dist_px = abs(hand_center[1] - line_pos)
            dist_cm = self.px_to_cm(dist_px)

            # Determine which side
            if hand_center[1] < line_pos:
                color = (255, 0, 0)  # Blue for above
            else:
                color = (0, 0, 255)  # Red for below

            # Draw vertical distance line
            cv2.line(frame,
                     (hand_center[0], hand_center[1]),
                     (hand_center[0], line_pos),
                     color, 2, cv2.LINE_AA)

            # Draw distance text
            mid_y = (hand_center[1] + line_pos) // 2
            text = f"{dist_cm:.1f}cm"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Background for text
            cv2.rectangle(frame,
                          (hand_center[0] - text_w // 2 - 5, mid_y - text_h - 5),
                          (hand_center[0] + text_w // 2 + 5, mid_y + 5),
                          (0, 0, 0), -1)

            cv2.putText(
                frame,
                text,
                (hand_center[0] - text_w // 2, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        else:
            # Distance to vertical line
            dist_px = abs(hand_center[0] - line_pos)
            dist_cm = self.px_to_cm(dist_px)

            if hand_center[0] < line_pos:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            cv2.line(frame,
                     (hand_center[0], hand_center[1]),
                     (line_pos, hand_center[1]),
                     color, 2, cv2.LINE_AA)

            mid_x = (hand_center[0] + line_pos) // 2
            text = f"{dist_cm:.1f}cm"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(frame,
                          (mid_x - text_w // 2 - 5, hand_center[1] - text_h - 10),
                          (mid_x + text_w // 2 + 5, hand_center[1] + 5),
                          (0, 0, 0), -1)

            cv2.putText(frame, text, (mid_x - text_w // 2, hand_center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Publish distance
        dist_msg = Float32()
        dist_msg.data = float(dist_cm)
        self.distance_pub.publish(dist_msg)

        return dist_cm

    def trigger_alert(self, hand_id, side):
        """Trigger crossing alert"""
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_alert_time > self.alert_cooldown:
            self.last_alert_time = current_time
            self.alert_active = True
            self.alert_message = f"🚨 HAND {self.crossing_count} CROSSED! ({side})"

            # Publish alert
            alert_msg = String()
            alert_msg.data = self.alert_message
            self.alert_pub.publish(alert_msg)

            # Publish crossing side
            side_msg = String()
            side_msg.data = side
            self.crossing_side_pub.publish(side_msg)

            self.get_logger().info(f"🚨 ALERT: {self.alert_message}")

    def draw_alert(self, frame):
        """Draw alert on frame if active"""
        if self.alert_active:
            current_time = time.time()
            elapsed = current_time - self.last_alert_time

            if elapsed < self.alert_duration:
                h, w = frame.shape[:2]

                # Semi-transparent overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h // 3), (w, 2 * h // 3), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                # Alert text
                text_size = cv2.getTextSize(self.alert_message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h // 2

                cv2.putText(frame, self.alert_message, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                # Flash effect
                if int(elapsed * 10) % 2 == 0:
                    cv2.putText(frame, "CROSSING DETECTED", (text_x, text_y - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                self.alert_active = False

    def check_crossing(self, hand_id, current_side, center, frame_size):
        """Check if hand crossed the line"""
        if hand_id in self.last_side:
            last_side = self.last_side[hand_id]

            if self.use_horizontal:
                line_pos = frame_size[0] * self.line_position
                current_pos = center[1]
            else:
                line_pos = frame_size[1] * self.line_position
                current_pos = center[0]

            # Check if crossed (changed side)
            if last_side != current_side:
                # Calculate crossing threshold
                dist_to_line = abs(current_pos - line_pos)

                if dist_to_line < self.threshold:
                    self.crossing_count += 1
                    dist_cm = self.px_to_cm(dist_to_line)

                    # Trigger alert
                    self.trigger_alert(hand_id, current_side)

                    self.get_logger().info(
                        f"✅ Hand {hand_id} crossed! Total: {self.crossing_count} (Distance: {dist_cm:.1f}cm)")

                    # Publish crossing count
                    msg = Int32()
                    msg.data = self.crossing_count
                    self.crossing_pub.publish(msg)

                    return True
        return False

    def process_frame(self, frame):
        """Process frame and return annotated frame"""
        h, w = frame.shape[:2]

        # Draw crossing line
        line_pos = self.draw_crossing_line(frame)

        # Find green dots (landmarks from gesture_detector)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours of green dots
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 5:  # Enough dots to form a hand
            # Calculate center of all green dots (hand position)
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

                # Find matching hand
                hand_id = self.find_matching_hand((center_x, center_y), frame.shape)

                if hand_id is None:
                    hand_id = self.next_hand_id
                    self.next_hand_id += 1
                    self.get_logger().info(f"🆕 New hand detected: {hand_id}")

                # Update tracking
                self.previous_centers[hand_id] = (center_x, center_y)

                # Get current side
                current_side = self.get_side((center_x, center_y), (h, w))

                # Draw hand center
                cv2.circle(frame, (int(center_x), int(center_y)), 8, (255, 0, 255), -1)

                # Draw hand ID with background
                id_text = f"Hand {hand_id}"
                (id_w, id_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame,
                              (int(center_x) + 10, int(center_y) - id_h - 5),
                              (int(center_x) + 10 + id_w + 10, int(center_y) + 5),
                              (0, 0, 0), -1)

                cv2.putText(frame, id_text, (int(center_x) + 15, int(center_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Draw distance to crossing line
                self.draw_distance_info(frame, (int(center_x), int(center_y)), line_pos)

                # Check for crossing
                self.check_crossing(hand_id, current_side, (center_x, center_y), (h, w))

                # Update last side
                self.last_side[hand_id] = current_side

        # Draw crossing count (prominently displayed)
        cv2.putText(frame, f"CROSSINGS: {self.crossing_count}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # Also draw smaller count at bottom for redundancy
        cv2.putText(frame, f"Count: {self.crossing_count}",
                    (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw alert if active
        self.draw_alert(frame)

        # Draw calibration info if in calibration mode
        if self.calibration_mode:
            if not self.calibration_complete:
                cv2.putText(frame, "CALIBRATION: Drag across a known-width object",
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Reference width: {self.reference_width_cm}cm",
                            (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                if len(self.calibration_points) == 1:
                    cv2.circle(frame, self.calibration_points[0], 5, (0, 255, 0), -1)

            cal_text = f"Cal: {self.pixels_per_cm:.2f} px/cm"
            cv2.putText(frame, cal_text, (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            original_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Resize frame to target display size
            frame = cv2.resize(original_frame, (self.display_width, self.display_height))

            # Process the frame (draw all annotations)
            processed_frame = self.process_frame(frame)

            # Show frame
            cv2.imshow("Crossing Counter", processed_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Toggle fullscreen with 'f' key
            if key == ord('f'):
                fullscreen = cv2.getWindowProperty("Crossing Counter",
                                                   cv2.WND_PROP_FULLSCREEN)
                if fullscreen == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty("Crossing Counter",
                                          cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Crossing Counter", self.display_width, self.display_height)
                else:
                    cv2.setWindowProperty("Crossing Counter",
                                          cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)

            # Increase size with '+' key
            elif key == ord('+') or key == ord('='):
                self.display_width = min(int(self.display_width * 1.2), self.max_width)
                self.display_height = min(int(self.display_height * 1.2), self.max_height)
                cv2.resizeWindow("Crossing Counter", self.display_width, self.display_height)
                self.get_logger().info(f"Window size: {self.display_width}x{self.display_height}")

            # Decrease size with '-' key
            elif key == ord('-') or key == ord('_'):
                self.display_width = max(int(self.display_width * 0.8), 640)
                self.display_height = max(int(self.display_height * 0.8), 480)
                cv2.resizeWindow("Crossing Counter", self.display_width, self.display_height)
                self.get_logger().info(f"Window size: {self.display_width}x{self.display_height}")

            # Reset count with 'r' key
            elif key == ord('r'):
                self.crossing_count = 0
                self.get_logger().info("🔄 Count reset to 0")

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