import autorootcwd
import json
import time
import os
import re
from src.indy_robot.Dxl_controller import DynamixelController
from src.indy_robot.indy_utils import indydcp_client as client

# Robot connection information
ROBOT_IP = "192.168.0.2"
ROBOT_NAME = "NRMK-Indy7"
indy = client.IndyDCPClient(ROBOT_IP, ROBOT_NAME)

# if not indy.connect():
#     print("Robot connection failed. Please check the IP address.")
#     exit()

# # Dynamixel motor controller connection
# dxl = DynamixelController()

# Directory where sequence JSON files are stored
WP_DIR = r"src\indy_robot\WP"

# Directory where sequence JSON files are stored
def get_sequence_files():
    sequence_files = [f for f in os.listdir(WP_DIR) if f.endswith(".json")]
    sequence_dict = {}
    pattern = re.compile(r"_(\d+)\.json$")  # Extract sequence number from filename
    
    for file in sequence_files:
        match = pattern.search(file)
        if match:
            sequence_number = int(match.group(1))
            sequence_dict[sequence_number] = os.path.join(WP_DIR, file)

    return sequence_dict

# Main loop
while True:
    sequence_dict = get_sequence_files()

    if not sequence_dict:
        print("No sequence files found. Please add JSON files to the directory.")
        time.sleep(5)
        continue

    print("\nAvailable sequence files:")
    for num in sorted(sequence_dict.keys()):
        print(f"  - Sequence {num}: {sequence_dict[num]}")

    try:
        selected_sequence = input("Enter the sequence number to execute (or type 'exit' to quit): ")

        if selected_sequence.lower() == "exit":
            print("Exiting program...")
            break

        selected_sequence = int(selected_sequence)

        if selected_sequence not in sequence_dict:
            print(f"Sequence {selected_sequence} does not exist.")
            continue

        WP_FILE = sequence_dict[selected_sequence]

        try:
            with open(WP_FILE, "r") as f:
                script_data = json.load(f)
        except FileNotFoundError:
            print(f"JSON file not found. Please check the path: {WP_FILE}")
            continue

        print(f"Robot and motor successfully connected. Executing selected sequence {selected_sequence} ({WP_FILE}).")

        for command in script_data["Program"]:
            cmd_type = command["cmd"]

            if cmd_type == "MoveJ":
                waypoints = command["waypoints"]
                for i, wp in enumerate(waypoints):
                    print(f"Executing MOVEJ [{i+1}] to position1... {wp}")
                    indy.set_joint_vel_level(2)
                    indy.joint_move_to(wp)
                    while not indy.get_robot_status()["movedone"]:
                        time.sleep(0.1)
                    print("Movement completed.")

            elif cmd_type == "MoveL":
                waypoints = command["waypoints"]
                for i, wp in enumerate(waypoints):
                    converted_wp = [wp[0], wp[1], wp[2]] + wp[3:]  # X, Y, Z 보존하고 나머지 추가
                    print(f"Executing MOVEL [{i+1}] to position... {converted_wp}")
                    indy.set_task_vel_level(2)
                    indy.task_move_to(converted_wp)
                    while not indy.get_robot_status()["movedone"]:
                        time.sleep(0.1)
                    print("Movement completed.")

            elif cmd_type == "MoveHome":
                home_position = [0, 0, -90, 0, -90, 0]
                print(f"Executing MOVEHOME to position... {home_position}")
                indy.set_joint_vel_level(2)
                indy.joint_move_to(home_position)
                while not indy.get_robot_status()["movedone"]:
                    time.sleep(0.1)
                print("Home position reached.")

            elif cmd_type == "Sleep":
                sleep_time = command["condition"]["time"]
                print(f"Waiting for {sleep_time} seconds...")
                time.sleep(sleep_time)
                print("Wait completed.")

            elif cmd_type == "GripperOpen":
                print("Executing Gripper Open operation.")
                dxl.Gripper_open()

            elif cmd_type == "Grippergrasp1":
                print("Executing Gripper Grasp1 operation.")
                dxl.Gripper_grasp1()

            elif cmd_type == "Grippergrasp2":
                print("Executing Gripper Grasp2 operation.")
                dxl.Gripper_grasp2()

            elif cmd_type == "GripperClose":
                print("Executing Gripper Close operation.")
                dxl.Gripper_close()

        print(f"Sequence {selected_sequence} execution completed.")

    except ValueError:
        print("Invalid input. Please enter a valid sequence number.")
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
        break

# Disconnect before exiting
indy.disconnect()
dxl.close()
print("System shutdown completed.")
