import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy

class CameraBridge(Node):
    def __init__(self):
        super().__init__('camera_bridge')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # --- CONFIGURATION ---
        # Source: The topic the camera publishes to
        self.camera_topic = '/camera/pose/sample'

        # Target: The topic MAVROS listens to for external position data
        self.mavros_topic = '/mavros/vision_pose/pose'
        # ---------------------

        # Create Subscriber
        self.subscription = self.create_subscription(
            Odometry,
            self.camera_topic,
            self.listener_callback,
            qos)

        # Create Publisher
        self.publisher = self.create_publisher(
            PoseStamped,
            self.mavros_topic,
            10)

        self.get_logger().info(f'Camera Bridge Started. Relaying {self.camera_topic} -> {self.mavros_topic}')

    def listener_callback(self, msg):
        # Ideally, Vicon and MAVROS usually both use ENU (East-North-Up) frames.
        # If so, we can simply pass the message through.

        # OPTIONAL: You might need to update the timestamp if there is significant
        # clock drift between the Vicon computer and the Jetson, 
        # but usually passing the original timestamp is safer for the EKF.
        pub_pose = PoseStamped()
        # pub_pose.header.frame_id = "odom"
        pub_pose.header = msg.header

        pub_pose.header.frame_id = "odom"        
        # pub_pose.pose = msg.pose.pose
        pub_pose.pose.position.x = -msg.pose.pose.position.x
        pub_pose.pose.position.y = -msg.pose.pose.position.y
        pub_pose.pose.position.z = 1.0 * msg.pose.pose.position.z
        
        pub_pose.pose.orientation.x = -msg.pose.pose.orientation.x
        pub_pose.pose.orientation.y = -msg.pose.pose.orientation.y
        pub_pose.pose.orientation.z = -msg.pose.pose.orientation.z
        pub_pose.pose.orientation.w = msg.pose.pose.orientation.w
        

        self.publisher.publish(pub_pose)
        # Uncomment below for debugging (it will be spammy)
        # self.get_logger().info('Relaying pose...')

def main(args=None):
    rclpy.init(args=args)
    camera_bridge = CameraBridge()
    rclpy.spin(camera_bridge)
    print('cp4')
    camera_bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
