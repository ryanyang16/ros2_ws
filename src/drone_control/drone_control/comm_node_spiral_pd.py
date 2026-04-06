import rclpy
import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, TwistStamped, Point, PoseArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
#from mavros_msgs.srv import CommandBool, SetMode

class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_5')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        
        # --- State Machine ---
        self.current_state = "INIT"
        self.got_initial_pose = False
        self.current_pose = PoseStamped()

        # --- Target Setpoints (Position) ---
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_orientation_w = 1.0
        self.target_orientation_x = 0.0
        self.target_orientation_y = 0.0
        self.target_orientation_z = 0.0

        # --- Target Setpoints (Velocity) ---
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0

        # --- MAVROS & Vicon Communication ---
        self.vision_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos)
        self.vicon_sub = self.create_subscription(PoseStamped, '/vicon/ROB498_Drone/ROB498_Drone', self.vicon_callback, 10)
        
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        
        # --- Vision Subscriber ---
        self.apriltag_sub = self.create_subscription(Point, '/vision/apriltag_error_2d', self.apriltag_callback, 10)
        self.target_2d = None
        self.last_apriltag_time = 0.0

        # --- Parameters for Centering & Landing ---
        self.kp_x = 0.005 
        self.kp_y = -0.005
        self.kp_z = -0.002
        self.ki_x = 0.0
        self.ki_y = -0.0
        self.kd_x = 0.0
        self.kd_y = -0.0
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.max_xy_speed = 0.3
        self.max_z_speed = 0.05
        self.descent_speed = -0.1
        self.drop_speed = -0.3

        # --- Services ---
        # launch
        self.srv_launch = self.create_service(Trigger, 'rob498_drone_5/comm/launch', self.callback_launch)
        # calibration wrt vicon to generate waypoints in camera frame that are within bounds of vicon
        self.srv_calib  = self.create_service(Trigger, 'rob498_drone_5/comm/calibrate', self.callback_calibrate)
        # test is: search (follows a pattern generated in calibration) + test which uses monocular camera for visual servoeing
        self.srv_test   = self.create_service(Trigger, 'rob498_drone_5/comm/test', self.callback_test) 
        # land
        self.srv_land_imx = self.create_service(Trigger, 'rob498_drone_5/comm/land_imx', self.callback_land_imx)
        # ignore these 2, just land in place
        self.srv_land   = self.create_service(Trigger, 'rob498_drone_5/comm/land', self.callback_land)
        self.srv_abort  = self.create_service(Trigger, 'rob498_drone_5/comm/abort', self.callback_abort)

        # Calibration Variables
        self.calib_waypoints = [
            [0.0, 0.0, 0.0],  # WP1: Replaced by current pos on trigger
            [0.5, 0.5, 0.5],  # WP2, etc downwards
            [0.5, 1.0, 1.0],
            [-0.5, 1.0, 0.5]
        ]
        self.calib_wp_index = 0
        self.calib_cam_pts = []
        self.calib_vic_pts = []
        self.current_vicon_pose = PoseStamped()
        
        self.is_collecting = False
        self.collection_start = 0.0
        self.temp_cam = []
        self.temp_vic = []
        
        self.R_vicon_to_cam = np.eye(3)
        self.t_vicon_to_cam = np.zeros(3)
        self.is_calibrated = False

        # Search Phase
        self.search_waypoints_cam = np.empty((0, 3))
        self.search_wp_index = 0
        self.waiting_at_wp = False
        self.wp_arrival_time = 0.0
        self.search_height = 1.5
        
        # Centering Phase
        self.land_automatically = False

        # The main code (runs at 50Hz)
        self.timer = self.create_timer(0.02, self.main_loop)
        self.get_logger().info("Unified Node Initialized. Awaiting /launch.")

    # ==========================================
    # HELPER FUNCTIONS
    # ==========================================
    def generate_spiral_pattern(self):
        """Generates an outward spiral in Vicon frame bounded by [-2, 2] and Z=1.5"""
        wps = []
        theta = 0.0
        d_theta = 0.5 # angle increment per step, in rad
        spacing = 0.4 # dist (meters) between spiral rings
        b = spacing / (2 * np.pi)
        
        while True:
            r = b * theta
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # set boundary constraint: xicon x,y in [-2.5, 2.5]
            if abs(x) > 2.0 or abs(y) > 2.0:
                break
                
            wps.append([x, y, self.search_height]) # z = 1.5 constraint, fly at a m above ground
            theta += d_theta
            
        return np.array(wps)

    def waypoints_to_cam_fr(self, vicon_waypoints):
        # Kabsch transformation for a set of waypoints
        local_waypoints = np.empty((0, 3))
        for wp in vicon_waypoints:
            local_wp = np.dot(self.R_vicon_to_cam, wp) + self.t_vicon_to_cam
            local_waypoints = np.vstack((local_waypoints, local_wp))
        return local_waypoints

    def compute_kabsch(self):
        # optimal rot and trans between vicon and camera frames
        P_vic = np.array(self.calib_vic_pts)
        P_cam = np.array(self.calib_cam_pts)

        centroid_v = np.mean(P_vic, axis=0)
        centroid_c = np.mean(P_cam, axis=0)

        v_centered = P_vic - centroid_v
        c_centered = P_cam - centroid_c

        H = np.dot(v_centered.T, c_centered)
        U, S, Vt = np.linalg.svd(H)
        R_mat = np.dot(Vt.T, U.T)

        if np.linalg.det(R_mat) < 0:
            Vt[2, :] *= -1
            R_mat = np.dot(Vt.T, U.T)

        t = centroid_c - np.dot(R_mat, centroid_v)

        self.R_vicon_to_cam = R_mat
        self.t_vicon_to_cam = t
        self.is_calibrated = True

        self.get_logger().info('\n=== KABSCH CALIBRATION COMPLETE ===')
        
        # create the spiral from 0,0,1.5 outward
        vicon_spiral_wps = self.generate_spiral_pattern()
        # absolute bounds into relative flying frame
        self.search_waypoints_cam = self.waypoints_to_cam_fr(vicon_spiral_wps)
        self.get_logger().info(f'Generated {len(self.search_waypoints_cam)} bounded search waypoints. Awaiting /test.')

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

    def vicon_callback(self, msg):
        self.current_vicon_pose = msg

    def apriltag_callback(self, msg):
        self.target_2d = msg
        self.last_apriltag_time = self.get_clock().now().nanoseconds / 1e9

    # --- Services ---
    def callback_launch(self, request, response):
        self.get_logger().info('Requested: LAUNCH')
        self.current_state = "LAUNCH"
        response.success = True
        return response

    def callback_calibrate(self, request, response):
        self.get_logger().info('Requested: CALIBRATE')
        self.current_state = "CALIBRATE"
        self.calib_waypoints[0] = [
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ]
        self.calib_wp_index = 0
        self.calib_cam_pts = []
        self.calib_vic_pts = []
        self.is_calibrated = False
        self.target_x = self.calib_waypoints[0][0]
        self.target_y = self.calib_waypoints[0][1]
        self.target_z = self.calib_waypoints[0][2]
        response.success = True
        return response

    def callback_test(self, request, response):
        if not self.is_calibrated:
            self.get_logger().warn("Warning: Proceeding with test without calibration")

        self.get_logger().info('Requested: TEST -> Initiating SPIRAL SEARCH.')
        self.current_state = "SEARCH"
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
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        tag_age = current_time - self.last_apriltag_time
        tag_visible = (self.target_2d is not None) and (tag_age < 0.5)
        current_z = self.current_pose.pose.position.z

        # --- POSITION CONTROL STATES ---
        if self.current_state == "INIT":
            self.target_x = self.current_pose.pose.position.x
            self.target_y = self.current_pose.pose.position.y
            self.target_z = self.current_pose.pose.position.z
            # P: Added target orientation initialization to (possible) address the tweak from launching
            self.target_orientation_w = self.current_pose.pose.orientation.w
            self.target_orientation_x = self.current_pose.pose.orientation.x
            self.target_orientation_y = self.current_pose.pose.orientation.y
            self.target_orientation_z = self.current_pose.pose.orientation.z

        elif self.current_state == "LAUNCH":
            self.target_z = 0.75 

        elif self.current_state == "CALIBRATE":
            if self.calib_wp_index < 4:
                current_pos = np.array([
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y,
                    self.current_pose.pose.position.z
                ])
                target_wp = np.array(self.calib_waypoints[self.calib_wp_index])
                distance = np.linalg.norm(current_pos - target_wp)

                if distance < 0.1 and not self.is_collecting:
                    self.is_collecting = True
                    self.collection_start = self.get_clock().now().nanoseconds / 1e9
                    self.temp_cam = []
                    self.temp_vic = []
                    self.get_logger().info(f"Arrived at Calib WP {self.calib_wp_index + 1}. Settling...")

                if self.is_collecting:
                    time_elapsed = current_time - self.collection_start
                    if 1.0 <= time_elapsed <= 2.5:
                        self.temp_cam.append([self.current_pose.pose.position.x, self.current_pose.pose.position.y, self.current_pose.pose.position.z])
                        if self.current_vicon_pose.pose.position.x != 0.0 or self.current_vicon_pose.pose.position.y != 0.0:
                            self.temp_vic.append([self.current_vicon_pose.pose.position.x, self.current_vicon_pose.pose.position.y, self.current_vicon_pose.pose.position.z])
                    
                    elif time_elapsed >= 3.0:
                        self.is_collecting = False
                        if len(self.temp_vic) == 0 or len(self.temp_cam) == 0:
                            self.get_logger().error("NO VALID DATA! Calibration invalid.")
                            self.current_state = "LAND"
                            return

                        self.calib_cam_pts.append(np.mean(self.temp_cam, axis=0))
                        self.calib_vic_pts.append(np.mean(self.temp_vic, axis=0))
                        self.calib_wp_index += 1
                        
                        if self.calib_wp_index < 4:
                            self.target_x = self.calib_waypoints[self.calib_wp_index][0]
                            self.target_y = self.calib_waypoints[self.calib_wp_index][1]
                            self.target_z = self.calib_waypoints[self.calib_wp_index][2]

            elif self.calib_wp_index == 4:
                self.compute_kabsch()
                self.calib_wp_index += 1 
                self.get_logger().info("Hovering at Calib WP 4. Awaiting /test to search.")

        elif self.current_state == "SEARCH":
            if not self.is_calibrated:
                self.search_waypoints_cam = self.generate_spiral_pattern()
                self.is_calibrated = True

            # target found -> transition to HOVER (TEST)
            if tag_visible:
                self.get_logger().info("TARGET SIGHTED! Intercepting -> Switching to Visual Servoing (TEST)")
                self.current_state = "TEST"
                return # Skip position control this loop
                
            # target not found -> continue spiral path
            if self.search_wp_index < len(self.search_waypoints_cam):
                target_wp = self.search_waypoints_cam[self.search_wp_index]
                self.target_x, self.target_y, self.target_z = target_wp[0], target_wp[1], target_wp[2]
                
                current_pos = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y, self.current_pose.pose.position.z])
                distance = np.linalg.norm(current_pos - target_wp)
                
                if not self.waiting_at_wp:
                    if distance < 0.2: # Arrived at spiral node
                        self.waiting_at_wp = True
                        self.wp_arrival_time = self.get_clock().now().nanoseconds / 1e9
                else:
                    if current_time - self.wp_arrival_time >= 0.5: # Brief pause at node
                        self.waiting_at_wp = False
                        self.search_wp_index += 1
            else:
                self.get_logger().info("Spiral complete. Target not found. Hovering at boundary limit.")

        elif self.current_state == "LAND":
            self.target_z = -0.1

        elif self.current_state == "ABORT":
            self.target_z = -0.1

        # --- VELOCITY CONTROL STATES ---
        elif self.current_state == "TEST":
            # state 2: HOVER
            if not tag_visible:
                self.vel_x, self.vel_y, self.vel_z = 0.0, 0.0, 0.0
                self.get_logger().warn("Tag lost! Holding velocity.", throttle_duration_sec=1.0)
                if self.get_clock().now().nanoseconds / 1e9 - self.last_apriltag_time > 2.0:
                    self.get_logger().warn("Can't Find tag. Resuming search.", throttle_duration_sec=1.0)
                    self.current_state = "SEARCH"
            else:
                alt = self.current_pose.pose.position.z

                v_x = self.target_2d.x * alt * self.kp_x
                v_y = self.target_2d.y * alt * self.kp_y
                v_z = (alt - self.search_height) * self.kp_z
                
                self.vel_x = max(min(v_x, self.max_xy_speed), -self.max_xy_speed)
                self.vel_y = max(min(v_y, self.max_xy_speed), -self.max_xy_speed)
                self.vel_z = max(min(v_z, self.max_z_speed), -self.max_z_speed) # 0.0 # Maintain height while centering

                if np.sqrt(self.target_2d.x**2 + self.target_2d.y**2) < 5 and self.land_automatically:
                    self.get_logger().warn("Beginning Landing Sequence", throttle_duration_sec=1.0)
                    self.current_state = "LAND_IMX"

        elif self.current_state == "LAND_IMX":
            # state 3: LAND_IMX
            if not tag_visible:
                if current_z < 0.5:
                    self.get_logger().fatal("Tag lost below 0.5m! Initiating blind DROP.", throttle_duration_sec=1.0)
                    self.vel_x, self.vel_y, self.vel_z = 0.0, 0.0, self.drop_speed
                else:
                    self.get_logger().warn("Tag lost high up! Pausing descent.", throttle_duration_sec=1.0)
                    self.vel_x, self.vel_y, self.vel_z = 0.0, 0.0, 0.0
                    if self.get_clock().now().nanoseconds / 1e9 - self.last_apriltag_time > 2.0:
                        self.get_logger().warn("Can't Find tag. Resuming search.", throttle_duration_sec=1.0)
                        self.current_state = "SEARCH"
            else:
                alt = self.current_pose.pose.position.z

                v_x = self.target_2d.x * alt * self.kp_x
                v_y = self.target_2d.y * alt * self.kp_y
                
                self.vel_x = max(min(v_x, self.max_xy_speed), -self.max_xy_speed)
                self.vel_y = max(min(v_y, self.max_xy_speed), -self.max_xy_speed)
                self.vel_z = self.descent_speed


        # ==========================================
        # VELOCITY OR POS PUBLISHER
        # ==========================================
        if self.current_state in ["TEST", "LAND_IMX"]:
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = 'base_link'
            
            twist_msg.twist.linear.x = float(self.vel_x)
            twist_msg.twist.linear.y = float(self.vel_y)
            twist_msg.twist.linear.z = float(self.vel_z)
            self.vel_pub.publish(twist_msg)
            
            # Keep position targets updated so it doesn't violently snap back
            # if switching back to a Position state
            self.target_x = self.current_pose.pose.position.x
            self.target_y = self.current_pose.pose.position.y
            self.target_z = self.current_pose.pose.position.z

        else:
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
