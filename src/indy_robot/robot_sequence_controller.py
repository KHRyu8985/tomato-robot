import json
import time
import os
import re
from src.indy_robot.Dxl_controller import DynamixelController
from src.indy_robot.indy_utils import indydcp_client as client

class RobotSequenceController:
    def __init__(self, robot_ip="192.168.0.2", robot_name="NRMK-Indy7", wp_dir="src/indy_robot/WP"):
        self.robot_ip = robot_ip
        self.robot_name = robot_name
        self.wp_dir = wp_dir
        self.indy = None
        self.dxl = None
        
    def connect(self):
        """connect to robot and gripper"""
        self.indy = client.IndyDCPClient(self.robot_ip, self.robot_name)
        if not self.indy.connect():
            raise ConnectionError("Robot connection failed")
            
        self.dxl = DynamixelController()
        return True
        
    def disconnect(self):
        """disconnect"""
        if self.indy:
            self.indy.disconnect()
        if self.dxl:
            self.dxl.close()
            
    def get_sequence_files(self):
        """get sequence file list"""
        sequence_files = [f for f in os.listdir(self.wp_dir) if f.endswith(".json")]
        sequence_dict = {}
        pattern = re.compile(r"_(\d+)\.json$")
        
        for file in sequence_files:
            match = pattern.search(file)
            if match:
                sequence_number = int(match.group(1))
                sequence_dict[sequence_number] = os.path.join(self.wp_dir, file)
                
        return sequence_dict
        
    def execute_sequence(self, sequence_number):
        """execute specific sequence"""
        sequence_dict = self.get_sequence_files()
        print(sequence_dict)
        
        if sequence_number not in sequence_dict:
            raise ValueError(f"Sequence {sequence_number} does not exist")
        else:
            print(f"Sequence {sequence_number} exists. Try to execute...")

        wp_file = sequence_dict[sequence_number]
        
        try:
            with open(wp_file, "r") as f:
                script_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {wp_file}")
            
        print(f"Executing sequence {sequence_number}")
        
        for command in script_data["Program"]:
            cmd_type = command["cmd"]
            
            if cmd_type == "MoveJ":
                # self._execute_movej(command["waypoints"])
                print(f"Executing MoveJ command... {command['waypoints']}")
            elif cmd_type == "MoveL":
                # self._execute_movel(command["waypoints"])
                print(f"Executing MoveL command... {command['waypoints']}")
            elif cmd_type == "MoveHome":
                # self._execute_movehome()
                print(f"Executing MoveHome command...")
            elif cmd_type == "Sleep":
                # self._execute_sleep(command["condition"]["time"])
                print(f"Executing Sleep command... {command['condition']['time']}")
            elif cmd_type == "GripperOpen":
                # self.dxl.Gripper_open()
                print(f"Executing GripperOpen command...")
            elif cmd_type == "Grippergrasp1":
                # self.dxl.Gripper_grasp1()
                print(f"Executing Grippergrasp1 command...")
            elif cmd_type == "Grippergrasp2":
                # self.dxl.Gripper_grasp2()
                print(f"Executing Grippergrasp2 command...")
            elif cmd_type == "GripperClose":
                # self.dxl.Gripper_close()
                print(f"Executing GripperClose command...")

        print(f"Sequence {sequence_number} execution completed")
        success = True

        return success
        
    def _execute_movej(self, waypoints):
        """execute MoveJ command"""
        for i, wp in enumerate(waypoints):
            print(f"Executing MOVEJ [{i+1}] to position... {wp}")
            self.indy.set_joint_vel_level(2)
            self.indy.joint_move_to(wp)
            while not self.indy.get_robot_status()["movedone"]:
                time.sleep(0.1)
                
    def _execute_movel(self, waypoints):
        """execute MoveL command"""
        for i, wp in enumerate(waypoints):
            converted_wp = [wp[0], wp[1], wp[2]] + wp[3:]
            print(f"Executing MOVEL [{i+1}] to position... {converted_wp}")
            self.indy.set_task_vel_level(2)
            self.indy.task_move_to(converted_wp)
            while not self.indy.get_robot_status()["movedone"]:
                time.sleep(0.1)
                
    def _execute_movehome(self):
        """execute MoveHome command"""
        home_position = [0, 0, -90, 0, -90, 0]
        print(f"Executing MOVEHOME to position... {home_position}")
        self.indy.set_joint_vel_level(2)
        self.indy.joint_move_to(home_position)
        while not self.indy.get_robot_status()["movedone"]:
            time.sleep(0.1)
            
    def _execute_sleep(self, sleep_time):
        """execute Sleep command"""
        print(f"Waiting for {sleep_time} seconds...")
        time.sleep(sleep_time) 