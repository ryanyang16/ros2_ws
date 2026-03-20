import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import TwistStamped, Point, PoseStamped
import math

class MissionCommander(Node):
    def __init__(self):
        super().__init__('mission_commander')

        # --- Mission States ---
        self.state = "SCAN"
        
        # --- Publishers & Subscribers ---
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        self.apriltag_sub = self.create_subscription(Point, '/vision/apriltag_error_2d', self.apriltag_callback, 10)
        self.local_pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, 10)

        # --- Services ---
        self.srv_land = self.create_service(Trigger, 'mission/execute_landing', self.execute_landing_callback)

        # --- Target Tracking Variables ---
        self.target_2d = None
        self.last_apriltag_time = 0.0
        self.execute_landing_flag = False
        
        self.current_local_pose = PoseStamped()
        self.got_local_pose = False

        # --- Hover & Descent Tuning ---
        self.descent_speed = -0.15 
        self.kp_hover_x = 0.005 
        self.kp_hover_y = 0.005

        # ==========================================
        # SCANNING VARIABLES (Uncomment the one you want)
        # ==========================================
        
        # Option A: SPIRAL VARIABLES
        self.scan_start_time = 0.0
        self.started_scanning = False
        self.spiral_forward_speed = 0.2
        
        # Option B: RELATIVE GRID VARIABLES
        # Assumes drone starts at [0,0] facing +X. 
        # Sweeps a 3m x 2m area in front and to the right of the drone.
        self.grid_waypoints = [
            [3.0, 0.0], [3.0, -0.5], 
            [0.0, -0.5], [0.0, -1.0], 
            [3.0, -1.0], [3.0, -1.5], 
            [0.0, -1.5], [0.0, -2.0],
            [3.0, -2.0]
        ]
        self.grid_wp_index = 0
        self.grid_kp = 0.5
        self.grid_max_speed = 0.3

        # Main Logic Loop (runs at 50Hz)
        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("Mission Commander Active. State: SCAN (Vicon Disabled)")

    def apriltag_callback(self, msg):
        self.target_2d = msg
        self.last_apriltag_time = self.get_clock().now().nanoseconds / 1e9

    def pose_callback(self, msg):
        self.current_local_pose = msg
        self.got_local_pose = True

    def execute_landing_callback(self, request, response):
        self.get_logger().info("Landing command received: DESCENT authorized.")
        self.execute_landing_flag = True
        response.success = True
        return response

    def control_loop(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link' 
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        tag_age = current_time - self.last_apriltag_time

        if self.state == "SCAN":
            
            # ==========================================
            # OPTION A: SPIRAL LOGIC
            # ==========================================
            if not self.started_scanning:
                self.scan_start_time = current_time
                self.started_scanning = True

            time_scanning = current_time - self.scan_start_time
            # Radius expands by 5cm every second
            current_radius = 0.5 + (0.05 * time_scanning)
            
            msg.twist.linear.x = self.spiral_forward_speed
            msg.twist.angular.z = self.spiral_forward_speed / current_radius

            # ==========================================
            # OPTION B: GRID LOGIC (Swap with Option A to test)
            # ==========================================
            """
            if not self.got_local_pose:
                return # Wait for RealSense

            if self.grid_wp_index < len(self.grid_waypoints):
                target_x = self.grid_waypoints[self.grid_wp_index][0]
                target_y = self.grid_waypoints[self.grid_wp_index][1]
                
                current_x = self.current_local_pose.pose.position.x
                current_y = self.current_local_pose.pose.position.y
                
                dx = target_x - current_x
                dy = target_y - current_y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 0.2:
                    self.grid_wp_index += 1
                    self.get_logger().info(f"Reached waypoint. Moving to index {self.grid_wp_index}")
                else:
                    v_x = dx * self.grid_kp
                    v_y = dy * self.grid_kp
                    
                    msg.twist.linear.x = max(min(v_x, self.grid_max_speed), -self.grid_max_speed)
                    msg.twist.linear.y = max(min(v_y, self.grid_max_speed), -self.grid_max_speed)
            else:
                msg.twist.linear.x = 0.0
                msg.twist.linear.y = 0.0
            """
            # ==========================================

            # Transition: Fresh AprilTag reading
            if self.target_2d is not None and tag_age < 0.5:
                self.state = "HOVER"
                self.started_scanning = False # Reset spiral if we lose it
                self.get_logger().info("AprilTag spotted! Transitioning to HOVER.")

        elif self.state == "HOVER":
            if tag_age > 1.0:
                self.state = "SCAN"
                self.get_logger().warn("Lost AprilTag! Reverting to SCAN.")
            else:
                # Hover P-Control
                v_x = self.target_2d.y * self.kp_hover_x
                v_y = self.target_2d.x * self.kp_hover_y
                
                msg.twist.linear.x = max(min(v_x, 0.3), -0.3)
                msg.twist.linear.y = max(min(v_y, 0.3), -0.3)
                
                if self.execute_landing_flag:
                    msg.twist.linear.z = self.descent_speed
                else:
                    msg.twist.linear.z = 0.0

        self.vel_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MissionCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
