import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageLogger(Node):
    def __init__(self):
        super().__init__('image_logger')
        # Change this to your RealSense RGB topic
        self.subscription = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        self.count = 0
        self.save_dir = "yolo_dataset/"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Timer to flag when to save an image (e.g., every 0.5 seconds)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.save_next = False

    def timer_callback(self):
        self.save_next = True

    def image_callback(self, msg):
        if self.save_next:
            try:
                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                filename = os.path.join(self.save_dir, f"target_{self.count:04d}.jpg")
                cv2.imwrite(filename, cv_image)
                self.get_logger().info(f"Saved: {filename}")
                self.count += 1
                self.save_next = False
            except Exception as e:
                self.get_logger().error(f"Failed to save image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
