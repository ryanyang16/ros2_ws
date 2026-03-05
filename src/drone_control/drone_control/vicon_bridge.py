import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy


class ViconBridge(Node):
    def __init__(self):
        super().__init__('vicon_bridge')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        # --- CONFIGURATION ---
        # Source: The topic Vicon publishes to
        self.vicon_topic = '/vicon/ROB498_Drone/ROB498_Drone'

        # Target: The topic MAVROS listens to for external position data
        self.mavros_topic = '/mavros/vision_pose/pose'
        # ---------------------

        # Create Subscriber
        self.subscription = self.create_subscription(
            PoseStamped,
            self.vicon_topic,
            self.listener_callback,
            qos)

        # Create Publisher
        self.publisher_ = self.create_publisher(
            PoseStamped,
            self.mavros_topic,
            10)

        self.get_logger().info(f'Vicon Bridge Started. Relaying {self.vicon_topic} -> {self.mavros_topic}')

    def listener_callback(self, msg):
        # Ideally, Vicon and MAVROS usually both use ENU (East-North-Up) frames.
        # If so, we can simply pass the message through.

        # OPTIONAL: You might need to update the timestamp if there is significant
        # clock drift between the Vicon computer and the Jetson, 
        # but usually passing the original timestamp is safer for the EKF.

        pub_pose = PoseStamped()
        pub_pose.header.stamp = self.get_clock().now().to_msg()
        pub_pose.header.frame_id = "odom"

        pub_pose.pose.position.x = msg.pose.pose.position.y
        pub_pose.pose.position.y = -msg.pose.pose.position.x
        pub_pose.pose.position.z = msg.pose.pose.position.z
        
        S = math.sqrt(0.5) #0.7071

        pub_pose.pose.orientation.x = S * (msg.pose.pose.orientation.x + msg.pose.pose.orientation.y)
        pub_pose.pose.orientation.y = S * (msg.pose.pose.orientation.y - msg.pose.pose.orientation.x)
        pub_pose.pose.orientation.z = S * (msg.pose.pose.orientation.z - msg.pose.pose.orientation.w)
        pub_pose.pose.orientation.w = S * (msg.pose.pose.orientation.w + msg.pose.pose.orientation.z)
        
        self.publisher.publish(pub_pose)
        # Uncomment below for debugging (it will be spammy)
        # self.get_logger().info('Relaying pose...')

def main(args=None):
    rclpy.init(args=args)
    vicon_bridge = ViconBridge()
    rclpy.spin(vicon_bridge)
    vicon_bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
