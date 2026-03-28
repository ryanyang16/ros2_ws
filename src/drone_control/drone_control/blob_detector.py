import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import numpy as np

class RedCircleDetector(Node):
    def __init__(self):
        super().__init__('red_circle_detector')
        
        # Subscribe to the IMX219 raw image topic published by imx219_node.py
        self.subscription = self.create_subscription(
            Image, 
            '/camera/bottom/image_raw', 
            self.image_callback, 
            10)
        
        # Publish pixel error from the center of the camera
        self.error_pub = self.create_publisher(Point, '/vision/red_circle_error_2d', 10)
        
        # Publish a debug video feed with bounding shapes drawn on it
        self.debug_pub = self.create_publisher(Image, '/vision/red_circle_debug', 10)
        
        self.bridge = CvBridge()

        # Camera center parameters (Assuming 640x480 resolution from your imx219 node)
        self.center_x = 320.0
        self.center_y = 240.0
        
        self.get_logger().info("Red Circle Detector Online. Waiting for frames...")

    def image_callback(self, msg):
        try:
            # 1. Convert ROS Image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 2. Convert to HSV color space for robust color filtering
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 3. Create masks for RED color
            # Red hue wraps around the 0 and 180 marks in OpenCV
            lower_red_1 = np.array([0, 120, 70])
            upper_red_1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

            lower_red_2 = np.array([170, 120, 70])
            upper_red_2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

            # Combine the two masks
            red_mask = mask1 | mask2

            # 4. Clean up the mask using morphological operations
            kernel = np.ones((5, 5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

            # 5. Find contours (blobs) in the binary mask
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_contour = None
            max_area = 0

            # 6. Filter contours by Area and Circularity
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter out small noise (300px is a safe baseline for a ~10-20cm target at 1.5m altitude)
                if area > 300:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0:
                        continue
                    
                    # Calculate circularity
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    
                    # A perfect circle has circularity == 1.0. We accept > 0.7 to allow for camera angle distortion
                    if circularity > 0.7 and area > max_area:
                        max_area = area
                        best_contour = contour

            # 7. Calculate center and publish error
            if best_contour is not None:
                # Calculate image moments to find the centroid of the blob
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Calculate pixel error from the center of the camera frame
                    error_msg = Point()
                    error_msg.x = float(cx) - self.center_x
                    error_msg.y = float(cy) - self.center_y
                    error_msg.z = 0.0 # Z is unused for 2D pixel error
                    
                    self.error_pub.publish(error_msg)

                    # Draw debug visuals (a green outline and blue center dot)
                    cv2.drawContours(frame, [best_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"Err: ({int(error_msg.x)}, {int(error_msg.y)})", 
                                (cx - 50, cy - 20), cv2.FONcenterT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 8. Publish the debug image back to ROS 2
            debug_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.debug_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f"CV Bridge / Image Processing Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RedCircleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()