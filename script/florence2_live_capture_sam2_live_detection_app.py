import autorootcwd
import torch
import gradio as gr
import cv2
from typing import Tuple, Optional
from gradio_webrtc import WebRTC
from PIL import Image
import numpy as np
import os
import time
from threading import Thread, Lock
import queue
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
from src.sam2_model import (
    setup_sam2_model,
    process_new_prompt,
    process_new_box,
    track_object,
)

# TODO: 현재 OD 태스크에만 사용 가능

SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bbox = []
point_coords = []
point_labels = []
initialized = False  # Keeps track of whether the model has been initialized

latest_frame = None  # Most recent frame received
processed_frame = None  # Most recent frame processed by the model
processing_thread = None
frame_lock = Lock()  # Ensures only one thread modifies processed_frame at a time
frame_queue = queue.Queue(maxsize=1)  # Only keep the latest frame
last_time = time.time()  # Last processed frame time

# Define the directory containing the images
image_dir = "data/pictures"

# Retrieve a list of image file paths (adjust the extensions as needed)
image_files = [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.endswith((".png", ".jpg", ".jpeg"))
]
image_files = image_files[:10]  # 10개만 사용

# Gradio expects examples as a list of lists where each inner list corresponds to the input components.
# For a single image input, each example is a one-element list
example_list = [[img] for img in image_files]


# 🏃 Background processing thread function (일부 프레임을 동적으로 스킵하고 처리)
def process_frame_in_thread():
    global latest_frame, processed_frame, frame_lock, point_coords, point_labels, initialized

    while True:
        start_time = time.time()
        frame = frame_queue.get()  # Get the latest frame from queue
        if frame is None:
            break  # Stop processing if None (exit signal)

        with (
            frame_lock
        ):  # Prevents multiple threads from modifying  processed_frame at the same time
            if not initialized and len(point_coords) > 0:
                # processed_frame = process_new_prompt(frame, point_coords, point_labels)/
                processed_frame = process_new_box(frame, bbox)
                initialized = True
            elif initialized:
                processed_frame = track_object(frame)
        frame_queue.task_done()  # Mark frame as processed, allowing the queue to remove it and move on to the next frame

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.4f} seconds")  # Log time


# Start the processing thread
processing_thread = Thread(
    target=process_frame_in_thread, daemon=True
)  # daemon=True flag makes sure the thread automatically stops when the main program exits.
processing_thread.start()


def clear_image_processing_components():
    return None, None, None, None, None


def on_mode_dropdown_change(mode):
    return [
        gr.Textbox(
            visible=mode == IMAGE_OPEN_VOCABULARY_DETECTION_MODE
        ),  # Open Vocabulary Detection은 프롬프트 입력이 필요
        gr.Textbox(
            label=(
                "Label Output"
                if mode == IMAGE_OPEN_VOCABULARY_DETECTION_MODE
                else "Caption Output"
            )  # Detection은 레이블 출력, Caption은 캡션 출력
        ),
    ]


def update_live_frame(frame):
    """
    This function processes each frame from the webcam stream,
    tracking objects that were previously detected.
    """
    global latest_frame
    latest_frame = frame

    if not frame_queue.full():
        frame_queue.put(frame)  # Add frame to queue

    return processed_frame if processed_frame is not None else frame


# 첫 프레임만 그대로 반환. 이후에는 queue에 저장된 프레임이 순차적으로 처리되기 때문에 processed_frame이 늘 존재함.


@torch.inference_mode()
@torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
def process_image(
    image_input, mode, text_input
) -> Tuple[
    Optional[Image.Image], Optional[str]
]:  # 연산의 결과로 PIL.Image.Image타입의 객체와 문자열을 반환 (둘 다 None값 가능)

    # Check if image_input is None or empty
    if image_input is None or (
        isinstance(image_input, np.ndarray) and image_input.size == 0
    ):
        gr.Info("Please upload an image.")
        return None, None

    if mode == IMAGE_OPEN_VOCABULARY_DETECTION_MODE:
        annotated_image, text_output, bbox_coordinates = run_open_vocabulary_detection(
            image_input, text_input
        )

    elif mode == IMAGE_CAPTION_GROUNDING_MODE:
        annotated_image, text_output, bbox_coordinates = run_caption_phrase_grounding(
            image_input, text_input
        )

    return annotated_image, text_output, bbox_coordinates


@torch.inference_mode()
@torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
def submit_prompt(text_input, mode=IMAGE_OPEN_VOCABULARY_DETECTION_MODE):
    """
    This function is used to process the video frame.
    It takes a frame as input and returns a processed frame.
    """
    global latest_frame, bbox, point_coords, point_labels, initialized
    if latest_frame is None:
        return None

    initialized = False

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
    processed_frame, text_output, bbox_coordinates = process_image(
        pil_image, mode, text_input
    )
    if bbox_coordinates and len(bbox_coordinates) > 0:

        bbox = bbox_coordinates  # 일단 사용 x
        x1, y1, x2, y2 = bbox_coordinates[0]
        point_coords = [[int((x1 + x2) / 2), int((y1 + y2) / 2)]]
        point_labels = [1]
        print(f"Starting SAM2 tracking at: {point_coords}")
        # 🚀 Start tracking immediately after detection!
        # processed_frame = process_new_prompt(latest_frame, point_coords, point_labels)
        # initialized = True

    return processed_frame, text_output, bbox_coordinates


css = """"""
# TODO: WebRTC 화질 조정하기
with gr.Blocks(css=css) as demo:
    # 이미지 탭
    with gr.Tab("Image"):

        image_processing_mode_dropdown_component = gr.Dropdown(
            choices=IMAGE_INFERENCE_MODES,
            value=IMAGE_INFERENCE_MODES[0],
            label="Mode",
            info="Select a mode to use.",
            interactive=True,
        )
        with gr.Row():
            with gr.Column():
                image_processing_image_input_component = gr.Image(
                    type="pil", label="Upload Image"
                )
                image_processing_text_input_component = gr.Textbox(
                    placeholder="Enter comma separated text prompts"
                )
                image_processing_clear_button_component = gr.Button(
                    value="Clear", variant="secondary"
                )
                image_processing_submit_button_component = gr.Button(
                    value="Submit", variant="primary"
                )
            with gr.Column():
                image_processing_image_output_component = gr.Image(
                    type="pil", label="Image Output"
                )
                image_processing_text_output_component = gr.Textbox(
                    label="Label Output", visible=True
                )
                image_processing_boundingbox_coordinates_output_component = gr.Textbox(
                    label="Bounding Box Coordinates", visible=True
                )

        gr.on(
            triggers=[
                image_processing_submit_button_component.click,
                image_processing_text_input_component.submit,
            ],
            fn=process_image,
            inputs=[
                image_processing_image_input_component,
                image_processing_mode_dropdown_component,
                image_processing_text_input_component,
            ],
            outputs=[
                image_processing_image_output_component,
                image_processing_text_output_component,
                image_processing_boundingbox_coordinates_output_component,
            ],
        )

        image_processing_clear_button_component.click(
            fn=clear_image_processing_components,
            outputs=[
                image_processing_image_input_component,
                image_processing_text_input_component,
                image_processing_image_output_component,
                image_processing_text_output_component,
                image_processing_boundingbox_coordinates_output_component,
            ],
        )
        image_processing_mode_dropdown_component.change(
            fn=on_mode_dropdown_change,
            inputs=image_processing_mode_dropdown_component,
            outputs=[
                image_processing_text_input_component,
                image_processing_text_output_component,
            ],
        )
        # Set examples
        # examples = gr.Gallery(value=image_files, columns=3, height='auto')
        examples = gr.Examples(
            examples=example_list, inputs=image_processing_image_input_component
        )
    # 웹캠 탭
    with gr.Tab("Video"):
        video_processing_mode_dropdown_component = gr.Dropdown(
            choices=IMAGE_INFERENCE_MODES,
            value=IMAGE_INFERENCE_MODES[0],
            label="Mode",
            info="Select a mode to use.",
            interactive=True,
        )
        with gr.Row(elem_classes="container"):
            with gr.Column(scale=1, min_width=400, elem_classes="webcam_column"):
                video_processing_stream_input_component = WebRTC(
                    label="Webcam",
                    rtc_configuration=None,
                    height=400,
                    width=480,
                    elem_classes="webrtc-video",
                )

            with gr.Column(scale=1, min_width=400):
                video_processing_captured_image_output_component = gr.Image(
                    label="Captured Image", type="pil", interactive=False
                )
                video_processing_text_output_component = gr.Textbox(
                    label="Label Output", visible=True
                )
                video_processing_boundingbox_coordinates_output_component = gr.Textbox(
                    label="Bounding Box Coordinates", visible=True
                )
        with gr.Row():
            with gr.Column():
                video_processing_text_input_component = gr.Textbox(
                    label="Prompt", placeholder="Enter comma-separated object names"
                )
                video_processing_capture_button_component = gr.Button(
                    "Submit", variant="primary"
                )

    # TODO: process_stream() 구현
    # examples.select(fn=load_example_image, inputs=[examples], outputs=[image_processing_image_input_component])
    video_processing_stream_input_component.stream(
        fn=update_live_frame,
        inputs=video_processing_stream_input_component,
        outputs=video_processing_stream_input_component,
    )
    gr.on(
        triggers=[
            video_processing_capture_button_component.click,
            video_processing_text_input_component.submit,
        ],
        fn=submit_prompt,
        inputs=[
            video_processing_text_input_component,
            video_processing_mode_dropdown_component,
        ],
        outputs=[
            video_processing_captured_image_output_component,
            video_processing_text_output_component,
            video_processing_boundingbox_coordinates_output_component,
        ],
    )

    video_processing_mode_dropdown_component.change(
        fn=on_mode_dropdown_change,
        inputs=video_processing_mode_dropdown_component,
        outputs=[
            video_processing_text_input_component,
            video_processing_text_output_component,
        ],
    )

if __name__ == "__main__":
    setup_florence_model(DEVICE)  # 모델 초기화
    setup_sam2_model(SAM2_CHECKPOINT, MODEL_CFG, DEVICE)
    demo.launch(share=True)
