import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import threading
import sys

class DummyVisionNode(Node):
    def __init__(self):
        super().__init__('dummy_vision')
        
        # Publishers matching the MissionCommander's expected topics
        self.yolo_pub = self.create_publisher(Point, '/vision/yolo_target_3d', 10)
        self.apriltag_pub = self.create_publisher(Point, '/vision/apriltag_error_2d', 10)
        
        # Initial fake coordinates
        self.yolo_pose = Point(x=5.0, y=0.0, z=0.0) # Start YOLO target 5 meters away
        self.apriltag_pose = Point(x=0.0, y=0.0, z=0.0) # Start AprilTag dead-center
        
        # Toggles to simulate the cameras gaining/losing sight
        self.publish_yolo = False
        self.publish_apriltag = False

        # Publish data at 10Hz
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Dummy Vision Node Ready.")

    def timer_callback(self):
        # Continually broadcast the fake coordinates if the toggles are on
        if self.publish_yolo:
            self.yolo_pub.publish(self.yolo_pose)
        if self.publish_apriltag:
            self.apriltag_pub.publish(self.apriltag_pose)

def keyboard_loop(node):
    """Runs a simple terminal menu to let you control the fake vision data"""
    print("\n--- Dummy Camera Controller ---")
    print("[1] YOLO: Spot Target (5m away)")
    print("[2] YOLO: Move Target Closer (-1m)")
    print("[3] YOLO: Lose Target (Simulate Handover Blindspot)")
    print("[4] AprilTag: Spot Target (Centered)")
    print("[5] AprilTag: Shift Target (Simulate Drone Drifting)")
    print("[0] Exit\n")
    
    while rclpy.ok():
        cmd = input("Enter command: ")
        
        if cmd == '1':
            node.publish_yolo = True
            node.yolo_pose.x = 5.0
            print(">> RealSense: Target spotted at 5.0m")
        elif cmd == '2':
            node.yolo_pose.x -= 1.0
            print(f">> RealSense: Target is now {node.yolo_pose.x}m away")
        elif cmd == '3':
            node.publish_yolo = False
            print(">> RealSense: Target LOST (Slipped under drone)")
        elif cmd == '4':
            node.publish_apriltag = True
            node.apriltag_pose.x = 0.0
            node.apriltag_pose.y = 0.0
            print(">> Monocular: AprilTag locked (0px error)")
        elif cmd == '5':
            node.apriltag_pose.x = 50.0 
            print(">> Monocular: AprilTag shifted (+50px X-axis drift)")
        elif cmd == '0':
            print("Exiting...")
            rclpy.shutdown()
            sys.exit(0)
        else:
            print("Invalid command.")

def main(args=None):
    rclpy.init(args=args)
    node = DummyVisionNode()
    
    # Run the keyboard input in a separate thread so it doesn't freeze the ROS 2 publishers
    thread = threading.Thread(target=keyboard_loop, args=(node,), daemon=True)
    thread.start()
    
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
