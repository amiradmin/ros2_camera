import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import serial


class GestureToArduino(Node):

    def __init__(self):
        super().__init__('gesture_to_arduino')

        # Adjust port if needed
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

        self.subscription = self.create_subscription(
            Int32,
            'gesture/number',
            self.callback,
            10
        )

        self.get_logger().info("Gesture → Arduino Bridge Started")

    def callback(self, msg):

        number = msg.data

        # Send number via serial
        self.ser.write(f"{number}\n".encode())

        self.get_logger().info(f"Sent to Arduino: {number}")


def main(args=None):
    rclpy.init(args=args)

    node = GestureToArduino()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ser.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
