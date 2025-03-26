import autorootcwd
import click
import cv2 as cv
import numpy as np

from src.yolo_model.yolov8_tomato_tracker import YOLOv8TomatoTracker
from src.hand_gesture.hand_tracker import HandTracker
from src.sam2_model.sam2_tracker import SAM2Tracker
from src.zed_sdk.zed_tracker import ZedTracker
from src.sam2_model.utils.mask import find_matching_tomato
from src.indy_robot.robot_sequence_controller import RobotSequenceController

@click.command()
@click.option('--camera', type=str, default='zed', help='camera type (zed, femto)')
@click.option('--show-feature', is_flag=True, default=False, help='visualization: mask decoder feature extraction result')
@click.option('--robot-control', is_flag=True, default=False, help='enable robot control')
def main(camera='zed', show_feature=False, robot_control=False):
    if camera == 'zed':
        zed_tracker = ZedTracker()
        
        if not zed_tracker.initialize_zed():
            print("[Error] failed to initialize ZED camera. Change to another camera.")
            cap = cv.VideoCapture(0)
            zed_tracker = None
            camera = 'femto'
    else:
        cap = cv.VideoCapture(0)
        zed_tracker = None

    yolo_tracker = YOLOv8TomatoTracker()
    sam2_tomato_tracker = SAM2Tracker(class_name="tomato")
    hand_tracker = HandTracker()
    sam2_tracker = SAM2Tracker()

    initial_frame = None
    
    if camera == 'zed':
        success, initial_frame, objects = zed_tracker.grab_frame_and_objects()
        if not success:
            print("[Error] failed to get initial frame from ZED camera")
            return
    else:
        ret, initial_frame = cap.read()
        if not ret:
            print("[Error] failed to get initial frame")
            return
            
    # Detect tomatoes using YOLO, and get the bounding boxes using SAM2
    detected_frame_yolo, tomato_boxes_buffer, yolo_results = yolo_tracker.detect_tomatoes(initial_frame)
    sam2_mask_image = None
    tomato_detection = None
    if tomato_boxes_buffer:
        print(f"[INFO] detected {len(tomato_boxes_buffer)} tomatoes using yolo")
        tomato_detection, sam2_mask_image = sam2_tomato_tracker.get_tomato_mask(initial_frame, tomato_boxes_buffer)
    else:
        print("[INFO] No tomatoes detected")

    # initialize robot controller (only when robot-control option is True)
    robot_controller = None
    robot_ready = False  # state of robot control ready
    if robot_control:
        try:
            robot_controller = RobotSequenceController()
            # robot_controller.connect()
            print("Robot controller initialized successfully")
        except Exception as e:
            print(f"Failed to initialize robot controller: {e}")
            robot_control = False

    while True:
        if camera == 'zed':
            success, frame, objects = zed_tracker.grab_frame_and_objects()
            if not success:
                continue

            zed_tracker.update_viewer()
            viewer_frame = zed_tracker.get_viewer_frame()
        else:
            ret, frame = cap.read()
            if not ret:
                print("[Error] failed to read frame")
                break
        
        debug_image = frame.copy()
        
        if sam2_mask_image is not None:
            cv.imshow("SAM2, YOLO Tomato Detection", sam2_mask_image)

        # hand gesture recognition
        debug_image, point_coords = hand_tracker.process_frame(frame, debug_image, None, None)
        
        # if new point_coords is detected, activate robot control
        if point_coords is not None:
            robot_ready = True
        
        if show_feature:
            # sam2 tracker with pca visualization
            debug_image, pca_visualization, has_valid_segment, new_mask = sam2_tracker.process_frame_with_visualization(frame, debug_image, point_coords)

            if new_mask is not None:
                matched_tomato_id, max_iou = find_matching_tomato(new_mask, tomato_detection, iou_threshold=0.8, debug=False, original_image=frame)

                if matched_tomato_id is not None:
                    cv.putText(debug_image, f"Matched: Tomato {matched_tomato_id} (IoU: {max_iou:.2f})", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                    
                    # robot control is enabled, robot is ready, and there is a matching tomato
                    if robot_control and robot_controller and robot_ready:
                        try:
                            success = robot_controller.execute_sequence(matched_tomato_id)
                            if success:
                                print(f"Successfully executed sequence {matched_tomato_id}")
                                robot_ready = False

                                print("[INFO] Re-detecting tomatoes using yolo...")
                                detected_frame_yolo, tomato_boxes_buffer, yolo_results = yolo_tracker.detect_tomatoes(frame)
                                if tomato_boxes_buffer:
                                    tomato_detection, sam2_mask_image = sam2_tomato_tracker.get_tomato_mask(frame, tomato_boxes_buffer)
                                    print(f"[INFO] detected {len(tomato_detection)} tomatoes")
                                else:
                                    print("[INFO] No tomatoes detected")
                                    tomato_detection = None
                                    sam2_mask_image = None
                            else:
                                print(f"Failed to execute sequence {matched_tomato_id}")
                        except Exception as e:
                            print(f"Failed to execute robot sequence: {e}")
                else:
                    cv.putText(debug_image, "No matching tomato", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            
            cv.imshow('hand gesture recognition', debug_image)

            if pca_visualization is not None:
                cv.imshow('PCA visualization', pca_visualization)
        else:
            # sam2 tracker
            debug_image, has_valid_segment, new_mask = sam2_tracker.process_frame(frame, debug_image, point_coords)
            
            if has_valid_segment and new_mask is not None and tomato_detection is not None:
                matched_tomato_id, max_iou = find_matching_tomato(new_mask, tomato_detection, iou_threshold=0.8, debug=False, original_image=frame)

                if matched_tomato_id is not None:
                    cv.putText(debug_image, f"Matched: Tomato {matched_tomato_id} (IoU: {max_iou:.2f})", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                    
                    # robot control is enabled, robot is ready, and there is a matching tomato
                    if robot_control and robot_controller and robot_ready:
                        try:
                            success = robot_controller.execute_sequence(matched_tomato_id)
                            if success:
                                print(f"Successfully executed sequence {matched_tomato_id}")
                                robot_ready = False

                                print("[INFO] Re-detecting tomatoes using yolo...")
                                detected_frame_yolo, tomato_boxes_buffer, yolo_results = yolo_tracker.detect_tomatoes(frame)
                                if tomato_boxes_buffer:
                                    tomato_detection, sam2_mask_image = sam2_tomato_tracker.get_tomato_mask(frame, tomato_boxes_buffer)
                                    print(f"[INFO] detected {len(tomato_detection)} tomatoes")
                                else:
                                    print("[INFO] No tomatoes detected")
                                    tomato_detection = None
                                    sam2_mask_image = None
                            else:
                                print(f"Failed to execute sequence {matched_tomato_id}")
                        except Exception as e:
                            print(f"Failed to execute robot sequence: {e}")
                else:
                    cv.putText(debug_image, "No matching tomato", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                    
            cv.imshow('hand gesture recognition', debug_image)

        key = cv.waitKey(10) & 0xFF
        if key == 27:  # ESC key terminates the program
            break
        elif key == 32: # space bar re-detects tomatoes using yolo
            print("[INFO] Re-detecting tomatoes using yolo...")
            detected_frame_yolo, tomato_boxes_buffer, yolo_results = yolo_tracker.detect_tomatoes(frame)
            if tomato_boxes_buffer:
                tomato_detection, sam2_mask_image = sam2_tomato_tracker.get_tomato_mask(frame, tomato_boxes_buffer)
                print(f"[INFO] detected {len(tomato_detection)} tomatoes")
            else:
                print("[INFO] No tomatoes detected")

    # when the program is terminated, clean up resources
    if robot_control and robot_controller:
        robot_controller.disconnect()
    
    if camera == 'zed':
        zed_tracker.close_zed()
    else:
        cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()