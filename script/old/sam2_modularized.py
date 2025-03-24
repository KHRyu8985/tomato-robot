import autorootcwd
import gradio as gr
from gradio_webrtc import WebRTC
from src.sam2_model import setup_sam2_model, process_new_prompt, track_object
import cv2
import numpy as np
import torch
import time
from threading import Thread, Lock
import queue

# sam2 결과를 gradio에서 보여주도록 하고, 실시간으로 입력받은 좌표를 활용하도록 하기
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bbox = []
point_coords = []
point_labels = []
initialized = False # Keeps track of whether the model has been initialized

latest_frame = None  # Most recent frame received
processed_frame = None  # Most recent frame processed by the model
processing_thread = None
frame_lock = Lock()  # Ensures only one thread modifies processed_frame at a time   
frame_queue = queue.Queue(maxsize=1) # Only keep the latest frame
last_time = time.time()  # Last processed frame time


# 🏃 Background processing thread function (일부 프레임을 동적으로 스킵하고 처리)
def process_frame_in_thread():
    global latest_frame, processed_frame, frame_lock, point_coords, point_labels, initialized

    while True:
        start_time = time.time()
        frame = frame_queue.get() # Get the latest frame from queue
        if frame is None:
            break # Stop processing if None (exit signal)

        with frame_lock: # Prevents multiple threads from modifying  processed_frame at the same time
            if not initialized and len(point_coords) > 0:
                processed_frame = process_new_prompt(frame, point_coords, point_labels)
                initialized = True
            elif initialized:
                processed_frame = track_object(frame)
        frame_queue.task_done() # Mark frame as processed, allowing the queue to remove it and move on to the next frame

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.4f} seconds") # Log time

# Start the processing thread
processing_thread = Thread(target=process_frame_in_thread, daemon=True)  # daemon=True flag makes sure the thread automatically stops when the main program exits.
processing_thread.start()


@torch.inference_mode()
@torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
def update_point_coords(x, y):
    global bbox, point_coords, point_labels, initialized
    initialized = False
    point_coords = [[x, y]]
    point_labels = [1]
    print(point_coords, point_labels)


def update_frame(frame):
    # start_time = time.time()
    global latest_frame, frame_queue, processed_frame
    latest_frame = frame # Update latest frame

    if not frame_queue.full():
        frame_queue.put(frame) # Add frame to queue

    return  processed_frame if processed_frame is not None else frame
# 첫 프레임만 그대로 반환. 이후에는 queue에 저장된 프레임이 순차적으로 처리되기 때문에 processed_frame이 늘 존재함.


with gr.Blocks() as demo:
    with gr.Row():
        stream = WebRTC(label="Camera", rtc_configuration=None)
    with gr.Row():
        x_coord = gr.Number(label="X", value=0)
        y_coord = gr.Number(label="Y", value=0)
    with gr.Row():
        # bbox_button = gr.Button("Set Bbox")
        point_button = gr.Button("Set Point")
    stream.stream(update_frame, inputs=stream, outputs=stream)
    # bbox_button.click(update_bbox, inputs=[x_coord, y_coord])
    point_button.click(update_point_coords, inputs=[x_coord, y_coord])

if __name__ == "__main__":
    setup_sam2_model(SAM2_CHECKPOINT, MODEL_CFG, DEVICE)
    demo.launch()
