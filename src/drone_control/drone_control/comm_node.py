import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy

class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_5')

        # BEST_EFFORT = ignore dropped packets and grab the latest ones, instead of hanging and waiting
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        self.current_state = "WAITING"
        
        # Keep track of where we are and where we want to go
        self.current_pose = PoseStamped()
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.got_initial_pose = False
        self.target_orientation_x = 0.0
        self.target_orientation_y = 0.0
        self.target_orientation_z = 0.0
        self.target_orientation_w = 1.0

        # Subscriber - pose from realsense, pos+orient, no velocity. Used for EKF on the cube
        self.vision_sub = self.create_subscription(PoseStamped, '/mavros/vision_pose/pose', self.callback, qos)

        # Publisher
        # TODO for tracking and smooth movement, might need to change to TwistStamped sent to /mavros/setpoint_velocity/cmd_vel
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        
        # Main Loop Timer - 50Hz
        self.timer = self.create_timer(0.02, self.main_loop)
        
        # Services
        self.srv_launch = self.create_service(Trigger, 'rob498_drone_5/comm/launch', self.callback_launch)  
        self.srv_test   = self.create_service(Trigger, 'rob498_drone_5/comm/test',   self.callback_test)    
        self.srv_land   = self.create_service(Trigger, 'rob498_drone_5/comm/land',   self.callback_land)    
        self.srv_abort  = self.create_service(Trigger, 'rob498_drone_5/comm/abort',  self.callback_abort)   

    def callback(self, msg):
        self.current_pose = msg
        
        # Lock in our starting position the first time we get Vicon data
        if not self.got_initial_pose:
            self.target_x = msg.pose.position.x
            self.target_y = msg.pose.position.y
            self.target_z = msg.pose.position.z
            # Pat:
            self.target_orientation_x = msg.pose.orientation.x
            self.target_orientation_y = msg.pose.orientation.y
            self.target_orientation_z = msg.pose.orientation.z
            self.target_orientation_w = msg.pose.orientation.w
            #
            self.got_initial_pose = True

    def callback_launch(self, request, response):
        self.get_logger().info('Launch commanded - Target Z set to 0.5m')
        self.current_state = "LAUNCH"
        # Lock in current X/Y so we go straight up, update Z to 0.5
        self.target_x = self.current_pose.pose.position.x
        self.target_y = self.current_pose.pose.position.y
        self.target_z = 0.5
        self.target_orientation_x = self.current_pose.pose.orientation.x
        self.target_orientation_y = self.current_pose.pose.orientation.y
        self.target_orientation_z = self.current_pose.pose.orientation.z
        self.target_orientation_w = self.current_pose.pose.orientation.w

        response.success = True
        return response

    def callback_test(self, request, response):
        self.get_logger().info('Test commanded - Hovering in place')
        self.current_state = "TEST"

        self.target_x = self.current_pose.pose.position.x
        self.target_y = self.current_pose.pose.position.y
        self.target_z = 0.5

        # TODO don't hard set orientation, keep the intial one like in launch?
        self.target_orientation_x = 0.0
        self.target_orientation_y = 0.0
        self.target_orientation_z = 0.0
        self.target_orientation_w = 1.0
        response.success = True
        return response

    def callback_land(self, request, response):
        self.get_logger().info('Land commanded - Target Z set to 0.0m')
        self.current_state = "LAND"
        # Maintain current X/Y, descend to floor
        self.target_x = self.current_pose.pose.position.x
        self.target_y = self.current_pose.pose.position.y
        self.target_z = 0.0
        response.success = True
        return response

    def callback_abort(self, request, response):
        self.get_logger().error('ABORT - Dropping to floor')
        self.current_state = "ABORT"
        self.target_z = 0.0
        response.success = True
        return response

    def main_loop(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        # TODO globally correct, but doesn't match the camera_bridge. Would we have to change later?
        msg.header.frame_id = 'map'
        
        # Continuously publish the target coordinates at 50Hz
        msg.pose.position.x = self.target_x
        msg.pose.position.y = self.target_y
        msg.pose.position.z = self.target_z
        
        # Valid quaternion prevents PX4 from rejecting the setpoint
        msg.pose.orientation.w = self.target_orientation_w
        msg.pose.orientation.x = self.target_orientation_x
        msg.pose.orientation.y = self.target_orientation_y
        msg.pose.orientation.z = self.target_orientation_z

        self.setpoint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
