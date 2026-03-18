import rclpy
import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, TwistStamped, Point, PoseArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
#from mavros_msgs.srv import CommandBool, SetMode

class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_5')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # State Machine
        self.current_state = "INIT"
        self.waypoints = np.empty((0, 3))
        self.waypoints_received = False
        self.got_initial_pose = False

        # Target Setpoints (For Position Control)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_orientation_w = 1.0
        self.target_orientation_x = 0.0
        self.target_orientation_y = 0.0
        self.target_orientation_z = 0.0
        self.current_pose = PoseStamped()

        # Target Setpoints (For Velocity Control)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0

        # AprilTag Tracking Variables
        self.target_2d = None
        self.last_apriltag_time = 0.0
        
        # Tuning Parameters for Centering & Landing
        self.kp_x = 0.001 
        self.kp_y = -0.001
        self.max_xy_speed = 0.3
        self.descent_speed = -0.15
        self.drop_speed = -0.4 

        # --- MAVROS Communication ---
        self.vision_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos)
        
        # We need BOTH publishers now
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        
        #self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        #self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # --- Vision Subscriber ---
        self.apriltag_sub = self.create_subscription(Point, '/vision/apriltag_error_2d', self.apriltag_callback, 10)

        # --- TA & Custom Services ---
        self.srv_launch = self.create_service(Trigger, 'rob498_drone_5/comm/launch', self.callback_launch)
        self.srv_test   = self.create_service(Trigger, 'rob498_drone_5/comm/test', self.callback_test)
        self.srv_land   = self.create_service(Trigger, 'rob498_drone_5/comm/land', self.callback_land)
        self.srv_abort  = self.create_service(Trigger, 'rob498_drone_5/comm/abort', self.callback_abort)
        
        # New Camera Landing Service
        self.srv_land_imx = self.create_service(Trigger, 'rob498_drone_5/comm/land_imx', self.callback_land_imx)
        self.sub_waypoints = self.create_subscription(PoseArray, 'rob498_drone_5/comm/waypoints', self.callback_waypoints, 10)

        # The main code (runs at 50Hz)
        self.timer = self.create_timer(0.02, self.main_loop)
        self.get_logger().info("Drone Node Initialized. Ready for AprilTag testing.")


    # ==========================================
    # CALLBACKS
    # ==========================================
    def pose_callback(self, msg):
        self.current_pose = msg
        if not self.got_initial_pose:
            self.target_x = msg.pose.position.x
            self.target_y = msg.pose.position.y
            self.target_z = msg.pose.position.z
            self.target_orientation_x = msg.pose.orientation.x
            self.target_orientation_y = msg.pose.orientation.y
            self.target_orientation_z = msg.pose.orientation.z
            self.target_orientation_w = msg.pose.orientation.w
            self.got_initial_pose = True

    def apriltag_callback(self, msg):
        self.target_2d = msg
        self.last_apriltag_time = self.get_clock().now().nanoseconds / 1e9

    def callback_waypoints(self, msg):
        pass 

    # --- Commands ---
    def callback_launch(self, request, response):
        self.get_logger().info('Requested: LAUNCH')
        self.current_state = "LAUNCH"
        response.success = True
        return response

    def callback_test(self, request, response):
        self.get_logger().info('Requested: TEST (AprilTag Centering)')
        self.current_state = "TEST"
        response.success = True
        return response

    def callback_land_imx(self, request, response):
        self.get_logger().info('Requested: LAND_IMX (Camera Descent)')
        self.current_state = "LAND_IMX"
        response.success = True
        return response

    def callback_land(self, request, response):
        self.get_logger().info('Requested: LAND (Standard)')
        self.current_state = "LAND"
        response.success = True
        return response

    def callback_abort(self, request, response):
        self.get_logger().fatal('Requested: ABORT')
        self.current_state = "ABORT"
        response.success = True
        return response


    # ==========================================
    # MAIN LOOP (50Hz)
    # ==========================================
    def main_loop(self):
        if not self.got_initial_pose:
            return
        #print("Hello World")
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        tag_age = current_time - self.last_apriltag_time
        tag_visible = (self.target_2d is not None) and (tag_age < 0.5)
        current_z = self.current_pose.pose.position.z
        # print("tag visible:", tag_visible)
        # --- POSITION CONTROL STATES ---
        if self.current_state == "INIT":
            self.target_x = self.current_pose.pose.position.x
            self.target_y = self.current_pose.pose.position.y
            self.target_z = self.current_pose.pose.position.z

        elif self.current_state == "LAUNCH":
            self.target_z = 1.5 

        elif self.current_state == "LAND":
            self.target_z = -0.1

        elif self.current_state == "ABORT":
            self.target_z = -0.1
            # arm_req = CommandBool.Request()
            # arm_req.value = False 
            #self.arm_client.call_async(arm_req)


        # --- VELOCITY CONTROL STATES ---
        elif self.current_state == "TEST":
            if not tag_visible:
                self.vel_x = 0.0
                self.vel_y = 0.0
                self.vel_z = 0.0
                self.get_logger().warn("Tag lost! Holding velocity.", throttle_duration_sec=1.0)
            else:
                
                v_x = self.target_2d.x * self.kp_x
                v_y = self.target_2d.y * self.kp_y
                print(v_x, v_y, self.target_2d.x, self.target_2d.y)
                
                self.vel_x = max(min(v_x, self.max_xy_speed), -self.max_xy_speed)
                self.vel_y = max(min(v_y, self.max_xy_speed), -self.max_xy_speed)
                self.vel_z = 0.0 # Hold altitude while centering

        elif self.current_state == "LAND_IMX":
            if not tag_visible:
                if current_z < 0.5:
                    self.get_logger().fatal("Tag lost below 0.5m! Initiating blind DROP.", throttle_duration_sec=1.0)
                    self.vel_x = 0.0
                    self.vel_y = 0.0
                    self.vel_z = self.drop_speed
                else:
                    self.get_logger().warn("Tag lost high up! Pausing descent.", throttle_duration_sec=1.0)
                    self.vel_x = 0.0
                    self.vel_y = 0.0
                    self.vel_z = 0.0
            else:
                v_x = self.target_2d.y * self.kp_x
                v_y = self.target_2d.x * self.kp_y
                
                self.vel_x = max(min(v_x, self.max_xy_speed), -self.max_xy_speed)
                self.vel_y = max(min(v_y, self.max_xy_speed), -self.max_xy_speed)
                self.vel_z = self.descent_speed


        # ==========================================
        # CONDITIONAL PUBLISHER
        # ==========================================
        if self.current_state in ["TEST", "LAND_IMX"]:
            # Publish Velocity Setpoints
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = 'base_link'
            
            twist_msg.twist.linear.x = float(self.vel_x)
            twist_msg.twist.linear.y = float(self.vel_y)
            twist_msg.twist.linear.z = float(self.vel_z)
            
            self.vel_pub.publish(twist_msg)
            
            # Keep position targets updated to current position so it doesn't snap 
            # wildly if you switch back to a Position state!
            self.target_x = self.current_pose.pose.position.x
            self.target_y = self.current_pose.pose.position.y
            self.target_z = self.current_pose.pose.position.z

        else:
            # Publish Position Setpoints
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'
            
            pose_msg.pose.position.x = float(self.target_x)
            pose_msg.pose.position.y = float(self.target_y)
            pose_msg.pose.position.z = float(self.target_z)
            
            pose_msg.pose.orientation.w = self.target_orientation_w
            pose_msg.pose.orientation.x = self.target_orientation_x
            pose_msg.pose.orientation.y = self.target_orientation_y
            pose_msg.pose.orientation.z = self.target_orientation_z

            self.setpoint_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
