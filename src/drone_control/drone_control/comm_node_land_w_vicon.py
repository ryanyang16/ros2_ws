import rclpy
import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, PoseArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from mavros_msgs.srv import CommandBool, SetMode
from scipy.spatial.transform import Rotation as R

class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_5')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        
        # ==========================================================
        # HARDCODE ZONE FOR PHASE 2 (LANDING PLATFORM)
        # ==========================================================
        # Set this to True after you paste your R and t values below
        self.USE_HARDCODED_TRANSFORM = True 
        
        if self.USE_HARDCODED_TRANSFORM:
            self.is_calibrated = True
            self.R_vicon_to_cam = np.array([[0.024896569211801833, 0.9996114347537488, -0.012535563450951217], [-0.9996117306644552, 0.024735720343708044, -0.012827004992996044], [-0.012511944672768758, 0.012850044693647686, 0.9998391508597156]])
            # PASTE YOUR PRINTED MATRICES HERE:
            #self.R_vicon_to_cam = np.array([
            #    [1.0, 0.0, 0.0],
            #    [0.0, 1.0, 0.0],
            #    [0.0, 0.0, 1.0]
            #])
            self.t_vicon_to_cam = np.array([-0.27865816415466305, -0.034079524195119046, -0.04330199271923668])
            # self.t_vicon_to_cam = np.array([0.0, 0.0, 0.0])
        else:
            self.is_calibrated = False
            self.R_vicon_to_cam = np.eye(3)
            self.t_vicon_to_cam = np.zeros(3)
        # ==========================================================

        # State Machine
        self.current_state = "INIT"
        self.waypoints = np.empty((0, 3))
        self.waypoints_camera = np.empty((0, 3)) 
        self.waypoints_received = False
        self.current_wp_index = 0
        self.has_launched = False
        self.waiting_at_wp = False
        self.wp_arrival_time = 0.0
        
        # Target Setpoints
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_orientation_w = 1.0
        self.target_orientation_x = 0.0
        self.target_orientation_y = 0.0
        self.target_orientation_z = 0.0
        
        self.current_pose = PoseStamped()
        self.got_initial_pose = False

        # MAVROS communication (drone and realsense)
        self.vision_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos)
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # Vicon frame transformation
        self.current_vicon_pose = PoseStamped()
        self.vicon_sub = self.create_subscription(PoseStamped, '/vicon/ROB498_Drone/ROB498_Drone', self.vicon_callback, 10)

        # TA grading communication
        self.srv_launch = self.create_service(Trigger, 'rob498_drone_5/comm/launch', self.callback_launch)
        self.srv_test   = self.create_service(Trigger, 'rob498_drone_5/comm/test', self.callback_test)
        self.srv_land   = self.create_service(Trigger, 'rob498_drone_5/comm/land', self.callback_land)
        self.srv_abort  = self.create_service(Trigger, 'rob498_drone_5/comm/abort', self.callback_abort)
        self.srv_calib  = self.create_service(Trigger, 'rob498_drone_5/comm/calibrate', self.callback_calibrate)
        self.sub_waypoints = self.create_subscription(PoseArray, 'rob498_drone_5/comm/waypoints', self.callback_waypoints, 10)

        # --- Calibration Waypoints ---
        self.calib_waypoints = [
            [-2.0, -2.0, 0.5], 
            [2.0, -2.0, 0.5],  
            [0.0, 0.0, 0.5],   
            [2.0, 2.0, 0.5],   
            [-2.0, 2.0, 0.5]  
            # (Truncated for brevity in the sample, you can put your full list back here)
        ]

        self.calib_wp_index = 0
        self.calib_cam_pts = []
        self.calib_vic_pts = []
        
        self.is_collecting = False
        self.collection_start = 0.0
        self.temp_cam = []
        self.temp_vic = []
        
        self.waypoints_transformed = False

        # The main code (runs at 50Hz)
        self.timer = self.create_timer(0.02, self.main_loop)
        self.get_logger().info("Drone Node Initialized. Hardcoded Transform: " + str(self.USE_HARDCODED_TRANSFORM))

    def compute_kabsch(self):
        """Computes optimal Rotation and Translation between Vicon and Camera frames."""
        P_vic = np.array(self.calib_vic_pts)
        P_cam = np.array(self.calib_cam_pts)

        centroid_v = np.mean(P_vic, axis=0)
        centroid_c = np.mean(P_cam, axis=0)

        v_centered = P_vic - centroid_v
        c_centered = P_cam - centroid_c

        H = np.dot(v_centered.T, c_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = centroid_c - np.dot(R, centroid_v)

        self.R_vicon_to_cam = R
        self.t_vicon_to_cam = t
        self.is_calibrated = True

        self.get_logger().info('\n\n=== KABSCH CALIBRATION COMPLETE ===')
        self.get_logger().info('COPY THE FOLLOWING LINES INTO YOUR HARDCODE ZONE:\n')
        self.get_logger().info(f'self.R_vicon_to_cam = np.array({repr(self.R_vicon_to_cam.tolist())})')
        self.get_logger().info(f'self.t_vicon_to_cam = np.array({repr(self.t_vicon_to_cam.tolist())})\n')
        self.get_logger().info('=== INITIATING AUTO-LANDING ===\n\n')

        # Automatically land so you can kill the script safely
        self.current_state = "LAND"

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

    def callback_waypoints(self, msg):
        pass # Ignored in this workflow since we track a moving platform instead of static waypoints

    # TA commands
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
        self.get_logger().info('Requested: TEST (Tracking Platform)')
        self.current_state = "TEST"
        response.success = True
        return response

    def callback_land(self, request, response):
        self.get_logger().info('Requested: LAND')
        self.current_state = "LAND"
        response.success = True
        return response

    def callback_abort(self, request, response):
        self.get_logger().fatal('Requested: ABORT')
        self.current_state = "ABORT"
        response.success = True
        return response

    def main_loop(self):
        if not self.got_initial_pose:
            return 

        if self.current_state == "INIT":
            self.target_x = self.current_pose.pose.position.x
            self.target_y = self.current_pose.pose.position.y
            self.target_z = self.current_pose.pose.position.z
            self.target_orientation_w = self.current_pose.pose.orientation.w
            self.target_orientation_x = self.current_pose.pose.orientation.x
            self.target_orientation_y = self.current_pose.pose.orientation.y
            self.target_orientation_z = self.current_pose.pose.orientation.z

        elif self.current_state == "LAUNCH":
            self.target_z = 0.5 

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
                    self.get_logger().info(f"Arrived at Calib WP {self.calib_wp_index + 1}. Settling for 1s...")

                if self.is_collecting:
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    time_elapsed = current_time - self.collection_start
                    
                    if 1.0 <= time_elapsed <= 2.5:
                        self.temp_cam.append([
                            self.current_pose.pose.position.x,
                            self.current_pose.pose.position.y,
                            self.current_pose.pose.position.z
                        ])
                        if self.current_vicon_pose.pose.position.x != 0.0 or self.current_vicon_pose.pose.position.y != 0.0:
                            self.temp_vic.append([
                                self.current_vicon_pose.pose.position.x,
                                self.current_vicon_pose.pose.position.y,
                                self.current_vicon_pose.pose.position.z
                            ])
                    
                    elif time_elapsed >= 3.0:
                        self.is_collecting = False
                        
                        if len(self.temp_vic) == 0 or len(self.temp_cam) == 0:
                            self.get_logger().error("NO VALID DATA RECEIVED! Landing.")
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

        elif self.current_state == "TEST":
            if not self.is_calibrated:
                self.get_logger().warn("Not calibrated! Cannot track platform. Go back to INIT.", throttle_duration_sec=2.0)
                return

            # 1. Grab raw platform position from Vicon
            p_vic = np.array([
                self.current_vicon_pose.pose.position.x,
                self.current_vicon_pose.pose.position.y,
                self.current_vicon_pose.pose.position.z
            ])

            # 2. Convert to Camera coordinate frame using your hardcoded matrix 
            # (Math: P_cam = R * P_vicon + t)
            p_cam = np.dot(self.R_vicon_to_cam, p_vic) + self.t_vicon_to_cam

            # 3. Get current drone position
            current_pos = np.array([
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                self.current_pose.pose.position.z
            ])

            # 4. Calculate error
            dx = p_cam[0] - current_pos[0]
            dy = p_cam[1] - current_pos[1]
            dist_xy = np.sqrt(dx**2 + dy**2)

            # --- DYNAMIC TRACKING LOGIC ---
            # Always match the X and Y of the moving platform
            self.target_x = p_cam[0]
            self.target_y = p_cam[1]

            if dist_xy > 0.2:
                # Approach phase: Hover 0.6m above the platform to avoid side collisions
                self.target_z = p_cam[2] + 0.5
                self.get_logger().info("TEST: Tracking platform (Hovering above)", throttle_duration_sec=1.0)
            else:
                # Descent phase: X and Y are locked in, lower Z to match the platform
                self.target_z = p_cam[2]
                self.get_logger().info("TEST: Locked on. Descending onto platform...", throttle_duration_sec=1.0)
                
                # Check if we have made contact (within 10cm of the platform's altitude)
                dz = p_cam[2] - current_pos[2]
                if abs(dz) < 0.1:
                    self.get_logger().info("Platform reached! Transitioning to LAND state.")
                    self.current_state = "LAND"

        elif self.current_state == "LAND":
            self.target_z = -0.1

        elif self.current_state == "ABORT":
            self.target_z = 0.0
            arm_req = CommandBool.Request()
            arm_req.value = False 
            self.arm_client.call_async(arm_req)

        # PUBLISH SETPOINTS
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.pose.position.x = float(self.target_x)
        msg.pose.position.y = float(self.target_y)
        msg.pose.position.z = float(self.target_z)
        
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
