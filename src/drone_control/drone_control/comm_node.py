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
        
        # State Machine
        self.current_state = "INIT"
        self.waypoints = np.empty((0, 3))
        self.waypoints_camera = np.empty((0, 3)) # NEW: Store transformed waypoints separately
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

        # --- Calibration Variables ---
        self.calib_waypoints = [
            [0.0, 0.0, 0.0],  # WP1: Will be replaced by current position
            [0.5, 0.5, 0.5],  # WP2
            [0.5, 1.0, 1.0],  # WP3
            [-0.5, 1.0, 0.5]  # WP4
        ]
        self.calib_wp_index = 0
        self.calib_cam_pts = []
        self.calib_vic_pts = []
        
        self.is_collecting = False
        self.collection_start = 0.0
        self.temp_cam = []
        self.temp_vic = []
        
        self.R_vicon_to_cam = np.eye(3)
        self.t_vicon_to_cam = np.zeros(3)
        self.is_calibrated = False
        self.waypoints_transformed = False

        # The main code (runs at 50Hz)
        self.timer = self.create_timer(0.02, self.main_loop)
        self.get_logger().info("Drone 5 Node Initialized and Waiting.")

    def waypoints_to_cam_fr(self, vicon_waypoints):
        """
        Applies the Kabsch-computed transformation to the TA's Vicon waypoints.
        """
        local_waypoints = np.empty((0, 3))
        
        for wp in vicon_waypoints:
            # P_cam = R * P_vicon + t
            local_wp = np.dot(self.R_vicon_to_cam, wp) + self.t_vicon_to_cam
            local_waypoints = np.vstack((local_waypoints, local_wp))
            
        return local_waypoints

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

        self.get_logger().info('\n=== KABSCH CALIBRATION COMPLETE ===')
        self.get_logger().info(f'\nR:\n{self.R_vicon_to_cam}')
        self.get_logger().info(f'\nt:\n{self.t_vicon_to_cam}')

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
        # Continually update our latest vicon pose for calibration sampling
        self.current_vicon_pose = msg

    def callback_waypoints(self, msg):
        if self.waypoints_received:
            return
        self.get_logger().info('Waypoints received from a TA')
        self.waypoints_received = True
        for pose in msg.poses:
            pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            self.waypoints = np.vstack((self.waypoints, pos))
            
        # If calibration finished BEFORE waypoints were sent, transform them right now.
        if self.is_calibrated and not self.waypoints_transformed:
            self.waypoints_camera = self.waypoints_to_cam_fr(self.waypoints)
            self.waypoints_transformed = True
            self.get_logger().info("Successfully transformed late-arriving TA Waypoints into Camera Frame!")

    # TA commands
    def callback_launch(self, request, response):
        self.get_logger().info('TA Requested: LAUNCH')
        self.current_state = "LAUNCH"
        response.success = True
        return response

    def callback_calibrate(self, request, response):
        self.get_logger().info('User Requested: CALIBRATE (4-Point Sequence)')
        self.current_state = "CALIBRATE"
        
        # WP1 is our current home position
        self.calib_waypoints[0] = [
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ]
        
        self.calib_wp_index = 0
        self.calib_cam_pts = []
        self.calib_vic_pts = []
        self.is_calibrated = False
        self.waypoints_transformed = False
        
        # Set target to Point 1 immediately
        self.target_x = self.calib_waypoints[0][0]
        self.target_y = self.calib_waypoints[0][1]
        self.target_z = self.calib_waypoints[0][2]
        
        response.success = True
        return response

    def callback_test(self, request, response):
        self.get_logger().info('TA Requested: TEST (Executing Waypoints)')
        self.current_state = "TEST"
        response.success = True
        return response

    def callback_land(self, request, response):
        self.get_logger().info('TA Requested: LAND')
        self.current_state = "LAND"
        response.success = True
        return response

    def callback_abort(self, request, response):
        self.get_logger().fatal('TA Requested: ABORT')
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
                # 1. Calculate Distance
                current_pos = np.array([
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y,
                    self.current_pose.pose.position.z
                ])
                target_wp = np.array(self.calib_waypoints[self.calib_wp_index])
                distance = np.linalg.norm(current_pos - target_wp)

                # 2. Check if arrived (10cm threshold)
                if distance < 0.1 and not self.is_collecting:
                    self.is_collecting = True
                    self.collection_start = self.get_clock().now().nanoseconds / 1e9
                    self.temp_cam = []
                    self.temp_vic = []
                    self.get_logger().info(f"Arrived at Calib WP {self.calib_wp_index + 1}. Settling for 1s...")

                # 3. Handle Timing and Collection
                if self.is_collecting:
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    time_elapsed = current_time - self.collection_start
                    
                    # Only collect data between 1.0 and 2.5 seconds
                    if 1.0 <= time_elapsed <= 2.5:
                        self.temp_cam.append([
                            self.current_pose.pose.position.x,
                            self.current_pose.pose.position.y,
                            self.current_pose.pose.position.z
                        ])
                        # Guard against missing vicon data
                        if self.current_vicon_pose.pose.position.x != 0.0 or self.current_vicon_pose.pose.position.y != 0.0:
                            self.temp_vic.append([
                                self.current_vicon_pose.pose.position.x,
                                self.current_vicon_pose.pose.position.y,
                                self.current_vicon_pose.pose.position.z
                            ])
                    
                    # 4. Finish Timer at 3.0s
                    elif time_elapsed >= 3.0:
                        self.is_collecting = False
                        
                        if len(self.temp_vic) == 0 or len(self.temp_cam) == 0:
                            self.get_logger().error("NO VALID DATA RECEIVED DURING SAMPLING WINDOW! Calibration invalid.")
                            self.current_state = "LAND"
                            return

                        # Average the tightly sampled data
                        self.calib_cam_pts.append(np.mean(self.temp_cam, axis=0))
                        self.calib_vic_pts.append(np.mean(self.temp_vic, axis=0))
                        self.get_logger().info(f"Calib WP {self.calib_wp_index + 1} recorded from tightly sampled window.")
                        
                        self.calib_wp_index += 1
                        
                        # Go to next WP
                        if self.calib_wp_index < 4:
                            self.target_x = self.calib_waypoints[self.calib_wp_index][0]
                            self.target_y = self.calib_waypoints[self.calib_wp_index][1]
                            self.target_z = self.calib_waypoints[self.calib_wp_index][2]
                            self.get_logger().info(f"Moving to Calib WP {self.calib_wp_index + 1}...")

            elif self.calib_wp_index == 4:
                self.compute_kabsch()
                
                # --- APPLY KABSCH TO TA WAYPOINTS IMMEDIATELY ---
                if self.waypoints_received and not self.waypoints_transformed:
                    self.waypoints_camera = self.waypoints_to_cam_fr(self.waypoints)
                    self.waypoints_transformed = True
                    self.get_logger().info("Successfully transformed TA Waypoints into Camera Frame!")
                elif not self.waypoints_received:
                    self.get_logger().warn("Calibration complete, but TA waypoints not yet received. Waiting...")

                self.calib_wp_index += 1 # Increment so we only compute once
                self.get_logger().info("Hovering at Calib WP 4. Ready for /test.")

        elif self.current_state == "TEST":

            # --- NORMAL NAVIGATION LOGIC (Using waypoints_camera) ---
            if self.waypoints_transformed and self.current_wp_index < len(self.waypoints_camera):
                target_wp = self.waypoints_camera[self.current_wp_index]
                
                self.target_x = target_wp[0]
                self.target_y = target_wp[1]
                self.target_z = target_wp[2]
                
                current_pos = np.array([
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y,
                    self.current_pose.pose.position.z
                ])
                
                distance = np.linalg.norm(current_pos - target_wp)
                
                if not self.waiting_at_wp:
                    if distance < 0.1:
                        self.get_logger().info(f"Reached Waypoint {self.current_wp_index + 1}/{len(self.waypoints_camera)}. Pausing for 3 seconds...")
                        self.waiting_at_wp = True
                        self.wp_arrival_time = self.get_clock().now().nanoseconds / 1e9
                else:
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    time_elapsed = current_time - self.wp_arrival_time
                    
                    if time_elapsed >= 3.0:
                        self.waiting_at_wp = False
                        self.current_wp_index += 1
                        
                        if self.current_wp_index < len(self.waypoints_camera):
                            self.get_logger().info(f"Proceeding to Waypoint {self.current_wp_index + 1}...")
                        else:
                            self.get_logger().info("Course complete! Holding final position.")
            
            elif self.current_wp_index >= len(self.waypoints_camera):
                 pass

        elif self.current_state == "LAND":
            self.target_z = -0.1

        elif self.current_state == "ABORT":
            self.target_z = 0.0
            arm_req = CommandBool.Request()
            arm_req.value = False 
            self.arm_client.call_async(arm_req)

        # ALWAYS PUBLISH SETPOINTS (50Hz)
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