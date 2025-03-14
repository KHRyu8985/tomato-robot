import autorootcwd
import cv2
import base64
import numpy as np
import threading
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from src.hand_gesture.hand_tracker import HandTracker
from src.sam2_model.sam2_tracker import SAM2Tracker

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

hand_tracker = HandTracker()
sam2_tracker = SAM2Tracker()

# Webcam thread running and Feature Visualization
thread_running = False
SHOW_FEATURE = False

def process_video():
    global thread_running, SHOW_FEATURE
    cap = cv2.VideoCapture(0)
    thread_running = True

    while thread_running:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to read frame")
            break

        debug_image = frame.copy()
        # Preprocessing with hand_tracker (update points, etc.)
        debug_image, point_coords = hand_tracker.process_frame(frame, debug_image, None, None)

        if SHOW_FEATURE:
            debug_image, pca_visualization = sam2_tracker.process_frame_with_visualization(frame, debug_image, point_coords)
            if pca_visualization is not None:
                _, buffer_feat = cv2.imencode('.jpg', pca_visualization)
                feature_base64 = base64.b64encode(buffer_feat).decode('utf-8')
                socketio.emit('feature_frame', {'image': feature_base64})
        else:
            debug_image = sam2_tracker.process_frame(frame, debug_image, point_coords)

        # Encode main debug video to base64 and send to client
        _, buffer = cv2.imencode('.jpg', debug_image)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'image': frame_base64})
        time.sleep(0.03)  # Approx. 30 FPS

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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
