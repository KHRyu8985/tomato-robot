import autorootcwd
import cv2
import click
import time

from src.hand_gesture.hand_tracker import HandTracker
from src.zed_sdk.zed_tracker import ZedTracker
from src.indy_robot.robot_sequence_controller import RobotSequenceController

@click.command()
@click.option('--camera', type=str, default='femto', help='camera type (zed, femto)')
@click.option('--mode', type=str, default='inference', help='mode (logging, inference)')
def main(camera='femto', mode='inference'):
    hand_tracker = HandTracker()
    robot_controller = RobotSequenceController()
    
    robot_lock = True
    current_point_coord = None
    last_nearest_tomato = None
    selection_start_time = None
    selected_tomato = None
    
    # initialize camera
    if camera == 'zed':
        zed_tracker = ZedTracker()
        if not zed_tracker.initialize_zed():
            print("[Error] failed to initialize ZED camera. Change to another camera.")
            cap = cv2.VideoCapture(0)
            zed_tracker = None
            camera = 'femto'
    else:
        cap = cv2.VideoCapture(0)
        zed_tracker = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # key input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l'):
            mode = 'logging'
            print("Changed to logging mode")
        elif key == 27:  # ESC
            mode = 'inference'
            hand_tracker.current_tomato_id = None
            print("Changed to inference mode")
        elif 49 <= key <= 52 and mode == 'logging' and current_point_coord is not None: # 1-4 key
            tomato_id = key - 48
            hand_tracker.save_tomato_coordinate(tomato_id, current_point_coord)
            print(f"Saved current point coordinates for tomato #{tomato_id} : {current_point_coord}")
        elif key == ord('q'):
            print("Quitting program")
            break

        debug_image = frame.copy()
        debug_image, _, _, landmark_list = hand_tracker.process_frame(frame, debug_image, number=None, key=key, use_point_tracker=False)

        if landmark_list is not None:
            current_point_coord = landmark_list[8]
            
            if mode == 'inference' and hand_tracker.prev_hand_gesture != "open":
                nearest_tomato, distance = hand_tracker.find_nearest_tomato(landmark_list[8])
                
                if nearest_tomato and distance < 100:
                    if last_nearest_tomato != nearest_tomato:
                        last_nearest_tomato = nearest_tomato
                        selection_start_time = time.time()
                        selected_tomato = None
                    else:
                        current_time = time.time()
                        if selection_start_time is not None:
                            selection_duration = current_time - selection_start_time
                            
                            if selection_duration < 1.5:
                                # first 1.5s to detect tomato
                                cv2.putText(debug_image, 
                                        f"Detecting tomato {nearest_tomato}: {1.5 - selection_duration:.1f}s", 
                                        (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            elif selection_duration < 4.5:  # 1.5~4.5s: waiting for confirmation
                                cv2.putText(debug_image, 
                                        f"Confirming tomato {nearest_tomato}: {4.5 - selection_duration:.1f}s", 
                                        (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            else:  # 4.5s or more: confirm selection
                                if selected_tomato != nearest_tomato:
                                    selected_tomato = nearest_tomato
                                    print(f"Selected tomato: {nearest_tomato}")
                                    try:
                                        print(f"Executing sequence {nearest_tomato}")
                                        success = robot_controller.execute_sequence(nearest_tomato)
                                        if success:
                                            print(f"Successfully executed sequence {nearest_tomato}")
                                    except Exception as e:
                                        print(f"Failed to execute robot sequence: {e}")
                                cv2.putText(debug_image, 
                                        f"Selected tomato: {nearest_tomato}", 
                                        (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    last_nearest_tomato = None
                    selection_start_time = None
                    selected_tomato = None
            else:  # hand is open or not in inference mode
                last_nearest_tomato = None
                selection_start_time = None
                selected_tomato = None
                if mode == 'inference':
                    cv2.putText(debug_image, "Ready to select new tomato", 
                              (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if selected_tomato is not None:
            tomato_key = f"tomato_{selected_tomato}"
            center = hand_tracker.tomato_coordinates[tomato_key]["center"]
            if center:
                cv2.circle(debug_image, (center["x"], center["y"]), 8, (0, 255, 0), -1)
                cv2.circle(debug_image, (center["x"], center["y"]), 12, (0, 255, 0), 2)
        
        mode_text = f"Mode: {mode.upper()}"
        cv2.putText(debug_image, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if mode == 'logging':   #logging mode
            cv2.putText(debug_image, 
                       "Press 1-4 to save current point, ESC for inference mode", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:   #inference mode
            cv2.putText(debug_image, 
                       "Press L to switch to logging mode", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Tomato Picker', debug_image)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()