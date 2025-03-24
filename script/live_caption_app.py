import autorootcwd
import torch
import gradio as gr
import cv2
import datetime
from typing import Tuple, Optional
from gradio_webrtc import WebRTC
from PIL import Image
from src.florence2_model import (
    setup_florence_model,
    run_open_vocabulary_detection,
    run_caption_phrase_grounding,
)
from src.florence2_model.modes import (
    IMAGE_INFERENCE_MODES,
    IMAGE_OPEN_VOCABULARY_DETECTION_MODE,
    IMAGE_CAPTION_GROUNDING_MODE,
)
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
latest_frame = None
latest_processed_frame = None
latest_text_output = None

auto_capture_running = True

def update_live_frame(frame):
    """
    This function processes each frame from the webcam stream,
    tracking objects that were previously detected.
    """
    global latest_frame
    latest_frame = frame
    return frame

@torch.inference_mode()
@torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
def process_frame(frame_list):
    """
    This function is used to process the video frame.
    It takes a frame as input and returns a processed frame.
    """
    global latest_frame, latest_processed_frame, latest_text_output
    if latest_frame is None:
        gallery_frame = [] if frame_list is None else frame_list
        return None, None, gallery_frame

    # Convert frame to PIL Image
    if isinstance(latest_frame, np.ndarray):
        # Check if the image is in BGR format (from OpenCV) and convert if needed
        if len(latest_frame.shape) == 3 and latest_frame.shape[2] == 3:
            # Convert BGR to RGB if necessary
            pil_image = Image.fromarray(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(latest_frame)
    else:
        pil_image = latest_frame
    # Run Florence2 first
    processed_frame, text_output, _ = run_caption_phrase_grounding(
        pil_image,""
    )
    latest_processed_frame = processed_frame
    latest_text_output = text_output
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # current_time = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_text = current_time+"\n\n"+text_output

    new_image = (processed_frame, formatted_text)
    if frame_list is None or not isinstance(frame_list, list):
        new_list = [new_image]
    else:
        if len(frame_list) > 19:
            new_list = frame_list[1:] + [new_image]
        else:
            new_list = frame_list + [new_image]
    return processed_frame, text_output, new_list

def show_gallery_caption(evt: gr.SelectData):
    frame = Image.open(evt.value['image']['path'])
    caption = evt.value['caption']
    return frame, caption

with gr.Blocks(theme='shivi/calm_seafoam') as demo:
    with gr.Tab("Detection"):
        with gr.Row():
            with gr.Column():
                image_output = gr.Image(label="Current Image", elem_id="my_image")
                caption_output = gr.Textbox(label="Caption")
            with gr.Column():
                selected_image = gr.Image(label="View Logs")
                selected_caption = gr.Textbox(label="Caption")
        with gr.Row():
            gallery = gr.Gallery(label="Gallery", columns=[6], rows=[1], object_fit="contain",height="400px",
                             allow_preview=False, show_fullscreen_button=False, elem_id="gallery-container")
            
    with gr.Tab("Camera"):
        stream = WebRTC(visible=True)
    timer = gr.Timer(3)

    stream.stream(update_live_frame, inputs=[stream], outputs=[stream])
    timer.tick(process_frame, inputs=gallery, outputs=[image_output, caption_output, gallery])
    gallery.select(show_gallery_caption, inputs=None, outputs=[selected_image, selected_caption])
if __name__ == "__main__":
    setup_florence_model(DEVICE)
    demo.launch()

