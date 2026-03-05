import rclpy
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

        self.publisher_.publish(msg)
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
