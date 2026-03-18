import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

class AprilTagDetector(Node):
    def __init__(self):
        super().__init__('apriltag_detector')
        
        # Subscribe to the IMX219 raw image topic
        self.subscription = self.create_subscription(Image, '/camera/bottom/image_raw', self.image_callback, 10)
        
        # Publish to the EXACT topic your MissionCommander is listening to
        self.error_pub = self.create_publisher(Point, '/vision/apriltag_error_2d', 10)
        
        # Publish a debug video feed with bounding boxes drawn on it
        self.debug_pub = self.create_publisher(Image, '/vision/apriltag_debug', 10)
        
        self.bridge = CvBridge()

        # Initialize the OpenCV Aruco detector for the tag25h9 dictionary
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_25h9)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        # SET YOUR TARGET ID HERE
        self.target_id = 5
        
        # Camera center parameters (Assuming 640x480 resolution)
        self.center_x = 320.0
        self.center_y = 240.0
        
        self.get_logger().info("AprilTag Detector Online. Looking for Tag25h9 ID: " + str(self.target_id))

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect all tags in the image
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                for i in range(len(ids)):
                    if ids[i][0] == self.target_id:
                        # Extract the 4 corners of our specific tag
                        c = corners[i][0]
                        
                        # Calculate the exact center pixel of the tag
                        tag_x = (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4.0
                        tag_y = (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4.0

                        # Calculate the pixel error from the center of the camera
                        error_msg = Point()
                        error_msg.x = tag_x - self.center_x
                        error_msg.y = tag_y - self.center_y
                        error_msg.z = 0.0 # Not used for 2D tracking
                        
                        self.error_pub.publish(error_msg)

                        # Draw a green bounding box and a red center dot for the debug stream
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                        cv2.circle(frame, (int(tag_x), int(tag_y)), 5, (0, 0, 255), -1)

            # Publish the debug image back to ROS 2
            debug_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.debug_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
