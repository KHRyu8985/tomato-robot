import autorootcwd
import torch
import gradio as gr
import cv2
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
def process_frame():
    """
    This function is used to process the video frame.
    It takes a frame as input and returns a processed frame.
    """
    global latest_frame
    if latest_frame is None:
        return None, None

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

    return processed_frame, text_output

css = """
#my_image img, 
#my_image .gr-image {
    width: 100% !important;
    height: 100% !important;
}
"""

with gr.Blocks(theme='shivi/calm_seafoam') as demo:
    with gr.Tab("Detection"):
        image_output = gr.Image(label="Status", elem_id="my_image")
        caption_output = gr.Textbox(label="Caption")
    with gr.Tab("Camera"):
        stream = WebRTC(visible=True)
    timer = gr.Timer(3)

    stream.stream(update_live_frame, inputs=[stream], outputs=[stream])
    timer.tick(process_frame, inputs=None, outputs=[image_output, caption_output])

if __name__ == "__main__":
    setup_florence_model(DEVICE)
    demo.launch()
