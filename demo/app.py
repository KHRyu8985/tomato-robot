import autorootcwd
import cv2
import base64
import numpy as np
import threading
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from src.hand_gesture.hand_tracker import HandTracker
from src.indy_robot.robot_sequence_controller import RobotSequenceController

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

hand_tracker = HandTracker()
robot_controller = RobotSequenceController()

thread_running = False
frame = None
robot_ready = False
robot_executing = False
last_hand_detection_time = time.time()
completion_message_start = None
showing_completion = False
mode = 'inference'
current_point_coord = None

@socketio.on('key_press')
def handle_key_press(data):
    global mode, current_point_coord
    key = data.get('key')
    
    if key == 'l':
        mode = 'logging'
        print("Changed to logging mode")
        socketio.emit('mode_change', {'mode': 'logging'})
    elif key == 'Escape':
        mode = 'inference'
        hand_tracker.current_tomato_id = None
        print("Changed to inference mode")
        socketio.emit('mode_change', {'mode': 'inference'})
    elif key in ['1', '2', '3', '4'] and mode == 'logging' and current_point_coord is not None:
        tomato_id = int(key)
        hand_tracker.save_tomato_coordinate(tomato_id, current_point_coord)
        print(f"Saved current point coordinates for tomato #{tomato_id} : {current_point_coord}")
        socketio.emit('coordinate_saved', {'tomato_id': tomato_id})

def process_video():
    global thread_running, frame, robot_ready, robot_executing, last_hand_detection_time, completion_message_start, showing_completion
    global mode, current_point_coord
    
    cap = cv2.VideoCapture(0)
    thread_running = True
    stream_paused = False
    last_nearest_tomato = None
    selection_start_time = None
    selected_tomato = None
    
    while thread_running:
        ret, frame = cap.read()
        if not ret:
            print("[Error] failed to read frame")
            break

        debug_image = frame.copy()
        current_time = time.time()

        if showing_completion:
            if current_time - completion_message_start <= 3.0:
                _, buffer = cv2.imencode('.jpg', debug_image)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'image': frame_base64})
                continue
            else:
                showing_completion = False
                completion_message_start = None
                selected_tomato = None
                robot_executing = False

        if not robot_executing:
            debug_image, _, _, landmark_list = hand_tracker.process_frame(frame, debug_image, None, None, mode='inference')
            
            if landmark_list is not None:
                current_point_coord = landmark_list[8]
                last_hand_detection_time = current_time
                robot_ready = True
                if stream_paused:
                    print("Hand detected, resuming stream")
                    stream_paused = False
                    socketio.emit('segment_status', {'detected': True, 'resumed': True})
                
                if mode == 'inference' and hand_tracker.prev_hand_gesture != "open":
                    nearest_tomato, distance = hand_tracker.find_nearest_tomato(landmark_list[8])
                    
                    if nearest_tomato and distance < 100:
                        if last_nearest_tomato != nearest_tomato:
                            last_nearest_tomato = nearest_tomato
                            selection_start_time = current_time
                            selected_tomato = None
                            socketio.emit('tomato_match_result', {'matched_id': nearest_tomato})
                        else:
                            if selection_start_time is not None:
                                selection_duration = current_time - selection_start_time
                                
                                if selection_duration < 1.5:
                                    socketio.emit('tomato_match_result', {
                                        'matched_id': nearest_tomato,
                                        'selection_time': selection_duration,
                                        'status': 'detecting'
                                    })
                                elif selection_duration < 4.5:
                                    socketio.emit('tomato_match_result', {
                                        'matched_id': nearest_tomato,
                                        'selection_time': selection_duration,
                                        'status': 'confirming'
                                    })
                                else:
                                    if selected_tomato != nearest_tomato:
                                        selected_tomato = nearest_tomato
                                        print(f"Selected tomato: {nearest_tomato}")
                                        if not robot_executing:
                                            socketio.emit('execute_robot_sequence', {'tomato_id': nearest_tomato})
                                    socketio.emit('tomato_match_result', {
                                        'matched_id': nearest_tomato,
                                        'selection_time': selection_duration,
                                        'status': 'selected'
                                    })
                    else:
                        last_nearest_tomato = None
                        selection_start_time = None
                        selected_tomato = None
                        socketio.emit('tomato_match_result', {'matched_id': None})
                else:
                    last_nearest_tomato = None
                    selection_start_time = None
                    selected_tomato = None
                    socketio.emit('tomato_match_result', {'matched_id': None})
                    
            elif current_time - last_hand_detection_time > 5.0 and not showing_completion:
                if not stream_paused:
                    print("No hand detected for 5 seconds, pausing stream")
                    socketio.emit('segment_status', {'detected': False, 'timeout': True})
                    socketio.emit('tomato_match_result', {'matched_id': None})
                    stream_paused = True
                continue

            if selected_tomato is not None:
                tomato_key = f"tomato_{selected_tomato}"
                center = hand_tracker.tomato_coordinates[tomato_key]["center"]
                if center:
                    cv2.circle(debug_image, (center["x"], center["y"]), 8, (0, 255, 0), -1)
                    cv2.circle(debug_image, (center["x"], center["y"]), 12, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', debug_image)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'image': frame_base64})
        
        time.sleep(0.01)

    if cap.isOpened():
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_stream')
def start_stream():
    global thread_running
    if not thread_running:
        thread = threading.Thread(target=process_video)
        thread.start()

@socketio.on('execute_robot_sequence')
def handle_robot_sequence(data):
    global frame, robot_ready, robot_executing, completion_message_start, showing_completion

    tomato_id = data.get('tomato_id')
    if tomato_id is not None and robot_controller and robot_ready and not robot_executing:
        try:
            robot_executing = True
            success = robot_controller.execute_sequence(tomato_id)
            if success:
                print(f"[INFO] Successfully executed sequence {tomato_id}")
                showing_completion = True
                completion_message_start = time.time()
                socketio.emit('robot_sequence_complete', {'tomato_id': tomato_id})
                robot_ready = False
        except Exception as e:
            print(f"[ERROR] Failed to execute robot sequence: {e}")
            robot_executing = False

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)