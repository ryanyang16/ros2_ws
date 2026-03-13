import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import TwistStamped, Point
import math

class MissionCommander(Node):
    def __init__(self):
        super().__init__('mission_commander')

        # --- Mission States ---
        self.state = "SCAN"
        
        # --- Publishers & Subscribers ---
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        self.yolo_sub = self.create_subscription(Point, '/vision/yolo_target_3d', self.yolo_callback, 10)
        self.apriltag_sub = self.create_subscription(Point, '/vision/apriltag_error_2d', self.apriltag_callback, 10)

        # landing trigger call
        self.srv_land = self.create_service(Trigger, 'mission/execute_landing', self.execute_landing_callback)

        # --- Target Tracking Variables ---
        self.target_3d = None
        self.target_2d = None
        self.last_yolo_time = 0.0
        self.last_apriltag_time = 0.0
        self.execute_landing_flag = False

        # --- Tuning Parameters ---
        self.max_approach_speed = 0.5  # meters per second
        self.descent_speed = -0.2      # meters per second (Z is up in ENU, so negative is down)
        self.follow_dist_x = 2.0       # stay 2m away from target when following
        
        # Proportional gains for the bottom camera pixel tracking
        self.kp_follow = 0.5        # for realsense
        self.kp_descent_x = 0.005   # for monocular cam
        self.kp_descent_y = 0.005

        # Main Logic Loop (runs at 50Hz)
        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("Mission Commander Active. State: SCAN")

    def yolo_callback(self, msg):
        self.target_3d = msg
        self.last_yolo_time = self.get_clock().now().nanoseconds / 1e9

    def apriltag_callback(self, msg):
        self.target_2d = msg
        self.last_apriltag_time = self.get_clock().now().nanoseconds / 1e9

    def execute_landing_callback(self, request, response):
        self.get_logger().info("Landing command received: LANDING")
        self.execute_landing_flag = True
        response.success = True
        return response


    def control_loop(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link' # Velocity commands are usually relative to the drone's body
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        yolo_age = current_time - self.last_yolo_time
        tag_age = current_time - self.last_apriltag_time

        # ==========================================
        # STATE: SCAN
        # ==========================================
        if self.state == "SCAN":
            # Fly forward slowly and yaw to scan the room
            msg.twist.linear.x = 0.2
            msg.twist.angular.z = 0.1
            
            # Transition: If we got a YOLO point in the last 1.0 second
            if self.target_3d is not None and (current_time - self.last_yolo_time) < 1.0:
                self.state = "APPROACH"
                self.get_logger().info("Target spotted! Transitioning to APPROACH.")

        # ==========================================
        # STATE: APPROACH
        # ==========================================
        elif self.state == "APPROACH":
            # 1. Calculate the 2D distance to the target
            dx = self.target_3d.x
            dy = self.target_3d.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance > 0:
                # 2. Normalize the vector and multiply by max speed
                msg.twist.linear.x = self.max_approach_speed * (dx / distance)
                msg.twist.linear.y = self.max_approach_speed * (dy / distance)
            
            # Transition: once we're closer than 2.5m, start tracking
            if distance <= 2.5:
                self.state = "FOLLOW"
                self.get_logger().info("Target locked, transitioning to TRACKING.")
            elif yolo_age > 2.0:
                # been more than 2s, lost the target. go back to scanning the room
                self.state = "SCAN"

        # ==========================================
        # STATE: FOLLOW / TRACK
        # ==========================================
        elif self.state == "FOLLOW":
            # want to be 2m away on x, centered on y
            error_x = self.target_3d.x - self.follow_dist_x
            error_y = self.target_3d.y

            # want to vary the speed based on how far we are
            v_x = error_x * self.kp_follow
            v_y = error_y * self.kp_follow

            # cap the max speed
            msg.twist.linear.x = max(min(v_x, self.max_approach_speed), -self.max_approach_speed)
            msg.twist.linear.y = max(min(v_y, self.max_approach_speed), -self.max_approach_speed)

            # wait for a landing command
            if self.execute_landing_flag:
                self.state = "HANDOVER"
                self.get_logger().info("Above the target, handing control over to monocular cam to land.")

        # ==========================================
        # STATE: HANDOVER
        # ==========================================
        elif self.state == "HANDOVER":
            # we were a bit behind the target, so move forwards till we can see it from monocular cam
            msg.twist.linear.x = 0.3
            msg.twist.linear.y = 0.0
            
            # Transition: As soon as the bottom camera gets a fresh AprilTag reading
            if self.target_2d is not None and tag_age < 0.5:
                self.state = "DESCENT"
                self.get_logger().info("AprilTag locked! Transitioning to DESCENT.")

        # ==========================================
        # STATE: DESCENT (Proportional Control)
        # ==========================================
        elif self.state == "DESCENT":
            # Use P-control to convert pixel error into velocity
            # NOTE: You may need to invert the signs (+/-) depending on how your camera is mounted!
            v_x = self.target_2d.y * self.kp_descent_x
            v_y = self.target_2d.x * self.kp_descent_y
            
            # Cap the maximum correction speed so the drone doesn't violently tilt
            msg.twist.linear.x = max(min(v_x, 0.3), -0.3)
            msg.twist.linear.y = max(min(v_y, 0.3), -0.3)
            
            # Command steady downward velocity
            msg.twist.linear.z = self.descent_speed
            
            # (Optional) Add a transition to LAND if Z-altitude is close to 0
            if msg.twist.linear.z < 0.5:
                self.target_z = -0.1

        self.vel_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MissionCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
