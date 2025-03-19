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

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

hand_tracker = HandTracker()
sam2_tracker = SAM2Tracker()

# Webcam thread running and Feature Visualization
thread_running = False
SHOW_FEATURE = False
SHOW_STREAM = False

# Camera type
CAMERA_TYPE = 'zed'

# Loading animation
def create_loading_animation(frame_idx, width=640, height=480):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Loading...", (width//2 - 150, height//2 - 50), 
               font, 0.8, (200, 200, 200), 2)
    
    return image

loading_frame_idx = 0

def process_video():
    global thread_running, SHOW_FEATURE, SHOW_STREAM, loading_frame_idx

    if CAMERA_TYPE == 'zed':
        zed_tracker = ZedTracker()
        if not zed_tracker.initialize_zed():
            return
    else:
        cap = cv2.VideoCapture(0)
        zed_tracker = None

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

        debug_image = frame.copy()
        debug_image, point_coords = hand_tracker.process_frame(frame, debug_image, None, None)
        
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
            
            if SHOW_FEATURE:
                debug_image, pca_visualization = sam2_tracker.process_frame_with_visualization(frame, debug_image, point_coords)
                has_valid_segment = pca_visualization is not None
                
                if pca_visualization is not None:
                    _, buffer_feat = cv2.imencode('.jpg', pca_visualization)
                    feature_base64 = base64.b64encode(buffer_feat).decode('utf-8')
                    socketio.emit('feature_frame', {'image': feature_base64})
            else:
                debug_image, has_valid_segment = sam2_tracker.process_frame(frame, debug_image, point_coords)
            
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

            _, buffer = cv2.imencode('.jpg', debug_image)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'image': frame_base64})

            if viewer_frame is not None:
                _, buffer_viewer = cv2.imencode('.jpg', viewer_frame)
                viewer_frame_base64 = base64.b64encode(buffer_viewer).decode('utf-8')
                socketio.emit('viewer_video_frame', {'image': viewer_frame_base64})
        
        # elif stream_paused:
        #     if time.time() % 0.1 < 0.03:
        #         loading_image = create_loading_animation(loading_frame_idx)
        #         loading_frame_idx += 1
                
        #         _, buffer = cv2.imencode('.jpg', loading_image)
        #         frame_base64 = base64.b64encode(buffer).decode('utf-8')
        #         socketio.emit('video_frame', {'image': frame_base64})
        
        time.sleep(0.03)

    socketio.emit('stream_stopped', {})
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

@socketio.on('stop_stream')
def stop_stream():
    global thread_running
    thread_running = False

@socketio.on('set_feature')
def set_feature(data):
    global SHOW_FEATURE
    SHOW_FEATURE = data.get('show_feature', False)
    print("SHOW_FEATURE set to", SHOW_FEATURE)

@click.command()
@click.option('--camera', default='zed', help='Camera type (zed, femto)')
def main(camera):

    global CAMERA_TYPE
    CAMERA_TYPE = camera

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
