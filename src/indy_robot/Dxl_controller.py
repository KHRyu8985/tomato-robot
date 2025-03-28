import autorootcwd
import time
import os
import json
from dynamixel_sdk import (
    PortHandler,
    PacketHandler
)

class DynamixelController:
    def __init__(self, port_name="COM3", baudrate=57600, dxl_id=1):
        """Dynamixel motor control class"""
        self.PORT_NAME = port_name
        self.BAUDRATE = baudrate
        self.DXL_ID = dxl_id
        self.PROTOCOL_VERSION = 2.0

        self.ADDR_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0
        self.EXTENDED_POSITION_MODE = 4  # Extended Position Mode value

        # JSON file path for motor positions
        self.POSITION_DIR = "src/indy_robot/"
        self.POSITION_FILE = os.path.join(self.POSITION_DIR, "motor_positions.json")

        # Initialize Dynamixel port
        self.portHandler = PortHandler(self.PORT_NAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        if self.portHandler.openPort():
            print(f"Dynamixel Port ({self.PORT_NAME}) opened successfully.")
            self.portHandler.setBaudRate(self.BAUDRATE)

            # Set Extended Position Mode
            print("Setting Extended Position Mode...")
            self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)  # Disable torque
            self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_OPERATING_MODE, self.EXTENDED_POSITION_MODE)  # Change mode
            self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)  # Enable torque
            print("Extended Position Mode has been set.")
        else:
            print(f"Failed to open Dynamixel Port ({self.PORT_NAME}).")
            exit()

        # Load settings from JSON file
        if not os.path.exists(self.POSITION_FILE):
            print(f"JSON file not found. Please check the path: {self.POSITION_FILE}")
            exit()

        with open(self.POSITION_FILE, "r") as file:
            positions = json.load(file)
            print(f"JSON file loaded successfully: {self.POSITION_FILE}")

        # Verify loaded position values
        self.open_pos = positions.get("open")
        self.grasp_pos1 = positions.get("grasp1")
        self.grasp_pos2 = positions.get("grasp2")
        self.close_pos = positions.get("close")

        if self.open_pos is None or self.close_pos is None:
            print("Position values in JSON file are not properly set.")
            exit()

        print(f"Loaded positions - Open: {self.open_pos}, Close: {self.close_pos}")

    def get_current_position(self):
        """Returns the current motor position"""
        pos, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_PRESENT_POSITION)
        return pos if pos < (1 << 31) else pos - (1 << 32)

    def move_motor(self, position):
        """Moves the Dynamixel motor to the specified position"""
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_GOAL_POSITION, position)
        time.sleep(1)  # Wait for movement stabilization
        present_position = self.get_current_position()
        print(f"Current position: {present_position}")

    def Gripper_open(self):
        """Moves to the Open position"""
        print("Moving to Open position.")
        self.move_motor(self.open_pos)

    def Gripper_grasp1(self):
        """Moves to the Grasp1 position"""
        print("Moving to Grasp1 position.")
        self.move_motor(self.grasp_pos1)

    def Gripper_grasp2(self):
        """Moves to the Grasp2 position"""
        print("Moving to Grasp2 position.")
        self.move_motor(self.grasp_pos2)

    def Gripper_close(self):
        """Moves to the Close position"""
        print("Moving to Close position.")
        self.move_motor(self.close_pos)

    def close(self):
        """Disconnects the motor"""
        self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
        self.portHandler.closePort()
        print("Motor system shutdown completed.")
