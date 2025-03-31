import autorootcwd
import cv2
import base64
import numpy as np
import threading
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import click

from src.hand_gesture.hand_tracker import HandTracker
from src.sam2_model.sam2_tracker import SAM2Tracker
from src.zed_sdk.zed_tracker import ZedTracker
from src.yolo_model.yolov8_tomato_tracker import YOLOv8TomatoTracker
from src.sam2_model.utils.mask import find_matching_tomato
from src.indy_robot.robot_sequence_controller import RobotSequenceController

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

hand_tracker = HandTracker()
sam2_tracker = SAM2Tracker()
yolo_tracker = YOLOv8TomatoTracker()
sam2_tomato_tracker = SAM2Tracker(class_name="tomato")
robot_controller = RobotSequenceController()

DETECT_TOMATOES = False
sam2_mask_image = None
tomato_detection = None

# Webcam thread running and Feature Visualization
thread_running = False
SHOW_FEATURE = False                                                                                                                       
SHOW_STREAM = False
CAMERA_TYPE = 'zed'

frame = None
robot_ready = False
robot_executing = False

# Loading animation
def create_loading_animation(frame_idx, width=640, height=480):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Loading...", (width//2 - 150, height//2 - 50), 
               font, 0.8, (200, 200, 200), 2)
    
    return image

loading_frame_idx = 0

def process_tomatoes(frame):
    global tomato_detection
    
    detected_frame_yolo, tomato_boxes_buffer, yolo_results = yolo_tracker.detect_tomatoes(frame)
    local_tomato_detection = None
    sam2_mask_image = None
    
    if tomato_boxes_buffer:
        print(f"[INFO] detected {len(tomato_boxes_buffer)} tomatoes using yolo")
        local_tomato_detection, sam2_mask_image = sam2_tomato_tracker.get_tomato_mask(frame, tomato_boxes_buffer)
        tomato_detection = local_tomato_detection
    else:
        print("[INFO] No tomatoes detected")

    return detected_frame_yolo, tomato_boxes_buffer, yolo_results, local_tomato_detection, sam2_mask_image

def process_video():
    global thread_running, SHOW_FEATURE, SHOW_STREAM, CAMERA_TYPE, loading_frame_idx
    global DETECT_TOMATOES, sam2_mask_image, tomato_detection, frame, robot_ready

    if CAMERA_TYPE == 'zed':
        zed_tracker = ZedTracker()
            
        if not zed_tracker.initialize_zed():
            print("[Error] failed to initialize ZED camera. Change to another camera.")
            cap = cv2.VideoCapture(0)
            zed_tracker = None
            CAMERA_TYPE = 'femto'
    else:
        cap = cv2.VideoCapture(0)
        zed_tracker = None

    initial_frame = None
    if CAMERA_TYPE == 'zed':
        success, initial_frame, objects = zed_tracker.grab_frame_and_objects()
        if not success:
            print("[Error] failed to get initial frame from ZED camera")
            return
    else:
        ret, initial_frame = cap.read()
        if not ret:
            print("[Error] failed to get initial frame")
            return
    
    # initialize robot controller (only when robot-control option is True)
    # if robot_controller.connect():
    #     print("[INFO] connected to robot controller")
    #     robot_control = True
    # else:
    #     print("[Error] failed to connect to robot controller")
    #     robot_control = False

    _, _, _, tomato_detection, sam2_mask_image = process_tomatoes(initial_frame)

    thread_running = True
    last_valid_segment_time = time.time()
    no_segment_timeout = 5.0  # 5 seconds timeout
    has_valid_segment_before = False
    stream_paused = False 
    
    while thread_running:
        if CAMERA_TYPE == 'zed':
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
            viewer_frame = None

        if DETECT_TOMATOES:
            print("[INFO] Detecting tomatoes on current frame...")
            _, _, _, tomato_detection, sam2_mask_image = process_tomatoes(frame)
            
            if sam2_mask_image is not None:
                _, buffer = cv2.imencode('.jpg', sam2_mask_image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('tomato_detection_result', {'image': image_base64})
                print("[INFO] Tomato detection result sent to client")

            DETECT_TOMATOES = False

        debug_image = frame.copy()
        if not robot_executing:
            debug_image, point_coords, expected_point_coords = hand_tracker.process_frame(frame, debug_image, None, None, use_point_tracker=True)
            if point_coords is not None:
                robot_ready = True
        
        if point_coords is not None:
            if stream_paused:
                print("Hand detected, resuming stream")
                stream_paused = False
                socketio.emit('segment_status', {'detected': True, 'resumed': True})
                
            SHOW_STREAM = True
            last_valid_segment_time = time.time()
            has_valid_segment_before = False    

        if SHOW_STREAM and not stream_paused:
            has_valid_segment = False
            matched_tomato_id = None
            
            if SHOW_FEATURE:
                debug_image, pca_visualization, has_valid_segment, new_mask = sam2_tracker.process_frame_with_visualization(frame, debug_image, point_coords)
               
                if new_mask is not None and tomato_detection:
                    matched_tomato_id, max_iou = find_matching_tomato(new_mask, tomato_detection, iou_threshold=0.8)
                
                if pca_visualization is not None:
                    _, buffer_feat = cv2.imencode('.jpg', pca_visualization)
                    feature_base64 = base64.b64encode(buffer_feat).decode('utf-8')
                    socketio.emit('feature_frame', {'image': feature_base64})
            else:
                debug_image, has_valid_segment, new_mask = sam2_tracker.process_frame(frame, debug_image, point_coords)
                
                if has_valid_segment and new_mask is not None and tomato_detection:
                    matched_tomato_id, max_iou = find_matching_tomato(new_mask, tomato_detection, iou_threshold=0.8)

            if robot_ready:
                socketio.emit('tomato_match_result', {'matched_id': matched_tomato_id})
            current_time = time.time()
            
            if has_valid_segment:
                last_valid_segment_time = current_time
                has_valid_segment_before = True
            
            no_segment_duration = current_time - last_valid_segment_time
            
            if no_segment_duration >= no_segment_timeout and has_valid_segment_before:
                print(f"No valid segment detected for {no_segment_duration:.1f} seconds. Pausing stream.")
                socketio.emit('segment_status', {'detected': False, 'timeout': True})
                stream_paused = True
                loading_frame_idx = 0
                continue

            if expected_point_coords is not None:
                x, y = int(expected_point_coords[0]), int(expected_point_coords[1])
                size = 7
                cv2.line(debug_image, (x - size, y), (x + size, y), color=(0, 0, 255), thickness=3)
                cv2.line(debug_image, (x, y - size), (x, y + size), color=(0, 0, 255), thickness=3)
                cv2.line(debug_image, (x - size, y), (x + size, y), color=(255, 255, 255), thickness=1)
                cv2.line(debug_image, (x, y - size), (x, y + size), color=(255, 255, 255), thickness=1)

            _, buffer = cv2.imencode('.jpg', debug_image)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'image': frame_base64})

            if viewer_frame is not None:
                _, buffer_viewer = cv2.imencode('.jpg', viewer_frame)
                viewer_frame_base64 = base64.b64encode(buffer_viewer).decode('utf-8')
                socketio.emit('viewer_video_frame', {'image': viewer_frame_base64})
        
        time.sleep(0.03)

    socketio.emit('stream_stopped', {})
    if CAMERA_TYPE != 'zed' and 'cap' in locals() and cap.isOpened():
        cap.release()
    if CAMERA_TYPE == 'zed' and 'zed_tracker' in locals():
        zed_tracker.close_zed()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_stream')
def start_stream():
    global thread_running
    if not thread_running:
        thread = threading.Thread(target=process_video)
        thread.start()

@socketio.on('stop_stream')
def stop_stream():
    global thread_running
    thread_running = False

@socketio.on('set_feature')
def set_feature(data):
    global SHOW_FEATURE
    SHOW_FEATURE = data.get('show_feature', False)
    print("SHOW_FEATURE set to", SHOW_FEATURE)

@socketio.on('detect_tomatoes')
def handle_detect_tomatoes():
    global DETECT_TOMATOES
    
    DETECT_TOMATOES = True

@socketio.on('execute_robot_sequence')
def handle_robot_sequence(data):
    global frame, robot_ready, robot_executing

    tomato_id = data.get('tomato_id')
    if tomato_id is not None and robot_controller and robot_ready and not robot_executing:
        try:
            robot_executing = True
            success = robot_controller.execute_sequence(tomato_id)
            if success:
                print(f"[INFO] Successfully executed sequence {tomato_id}")
                _, _, _, tomato_detection, sam2_mask_image = process_tomatoes(frame)
                socketio.emit('robot_sequence_complete', {'tomato_id': tomato_id})
                robot_ready = False
                robot_executing = False
            else:
                print(f"[ERROR] Failed to execute sequence {tomato_id}")
        except Exception as e:
            print(f"[ERROR] Failed to execute robot sequence: {e}")

@click.command()
@click.option('--camera', default='zed', help='Camera type (zed, femto)')
def main(camera):
    global CAMERA_TYPE
    CAMERA_TYPE = camera
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
