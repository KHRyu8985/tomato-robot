import gradio as gr
import torch 
from gradio_webrtc import WebRTC
from PIL import Image
import cv2
from typing import Tuple, Dict, Optional, Union, Any
from src.florence2_model.florence import load_florence_model, run_florence_inference,\
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_OPEN_VOCABULARY_TASK
from src.florence2_model.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MODE
import supervision as sv
from huggingface_hub import hf_hub_download # for downloading YOLOv10 model
from src.yolo_model.yolov10 import YOLOv10

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
yolo_model_file = hf_hub_download(
    repo_id="onnx-community/yolov10n", filename="onnx/model.onnx"
)
YOLO_MODEL = YOLOv10(yolo_model_file)

# Manually enters a context manager that automatically casts CUDA operations to use bfloat16 precision. (Using bfloat16 can reduce memory usage and improve computation speed without the full loss of accuracy you might get from lower-precesion computations.
# This is similar to the common use of FP16 autocast but tailored for bfloat16, which is particularly useful on hardware that supports it.)
torch.autocast(device_type = DEVICE, dtype = torch.bfloat16).__enter__()
# Enabling TF32 on Supported GPUs
if torch.cuda.get_device_properties(0).major >=8: # This checks the GPU's compute capability. A major version of the 8 or higher indicates an Nvidia Ampere (or later) GPU.
    torch.backends.cuda.matmul.allow_tf32 = True # If the condition is met, it enables TensorFloat-32 (TF32) support for both CUDA's matrix multiplication (matmul)
    torch.backends.cudnn.allow_tf32 = True
    # TF32 is a math mode that allows the GPU to perform matrix multiplications and convolutions faster while still maintaining acceptable precision for deep learning tasks.
    # Enabling TF32 on supported GPUs can lead to significant performance improvements.

"""
    Further Clarification on TF32 and bfloat16:

    TF32 (TensorFloat-32) 
    - This not a separate data type but a math mode available on Nvidia Ampere (and later) GPUs.
    - TF32 uses standard FP32 storage but performs matrix multiplications with a reduced mantissa precission (about 10 bits instead of 23)
    - The goal is to speed up computations while still providing reasonable precision for deep learning tasks.

    bfloat16(Brain Floating Point 16):
    - This is a distinct 16-bit floating point data type.
    - It maintains the same exponent range as FP32(8bits) but uses only 7 bits for the mantissa
    - bfloat16 reduced memory usage and can accelerate computations, but with lower precision compared to FP32 or TF32's intermediate math mode.

    While both are used to speed up computations and reduce resource usage, TF32 is a performance mode for FP32 operations, whereas bfloat16 is an actual lower-precision data type.
"""

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS) # If no argument is provided, it uses the default palette. (sv.ColorPalette())
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.TOP_LEFT,
    text_color=sv.Color.from_hex("#000000")
)

def clear_image_processing_components():
    return None, None, None, None, None

def on_mode_dropdown_change(mode):
    return[
        gr.Textbox(visible= mode==IMAGE_OPEN_VOCABULARY_DETECTION_MODE), # 프롬프트 입력이 필요하다
        gr.Textbox(visible= mode==IMAGE_CAPTION_GROUNDING_MODE), # Caption은 input이 필요없고 output은 있어야 한다. 
    ]

def annotate_image(image, detections):
    output_image = image.copy()
    output_image = BOX_ANNOTATOR.annotate(scene=image, detections=detections)
    output_image = LABEL_ANNOTATOR.annotate(scene=output_image, detections=detections)

    return output_image

def convert_to_od_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts the dictionary with 'bboxes' and 'bboxes_labels' into a dictionary
    with separate 'bboxes' and 'labels' keys.

    Parameters:
    - data: The input dictionary containing 'bboxes' and 'bboxes_labels', 'polygons' and 'polygons_labels' keys.
            (The result of the '<OPEN_VOCABULARY_DETECTION>' task)
    Returns:
    - A dictionary with separate 'bboxes' and 'labels' keys formatted for object detection results.
    """ 
    bboxes = data.get('bboxes', []) # data.get(key, default) -> returns value associated with key if it exsists, otherwise returns default
    labels = data.get('bboxes_labels', [])

    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }
    return od_results
    return frame

@torch.inference_mode()
@torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
def process_image(
        image_input, mode, text_input
    ) -> Tuple[Optional[Image.Image], Optional[str]]: # 연산의 결과로 PIL.Image.Image타입의 객체와 문자열을 반환 (둘 다 None값 가능)

    # 이미지 입력이 없으면 즉시 종료
    if not image_input:
        gr.Info("Please upload an image.")
        return None, None

    if mode == IMAGE_OPEN_VOCABULARY_DETECTION_MODE:
        task = FLORENCE_OPEN_VOCABULARY_TASK
        _, result = run_florence_inference(
            FLORENCE_MODEL, 
            FLORENCE_PROCESSOR, 
            DEVICE, 
            image_input, 
            task, 
            text_input)
        
        converted_result = convert_to_od_format(result[task]) # convert to OD

        text_output = ",".join(converted_result['labels'])
        bbox_coordinates = ""
        for i in range(len(converted_result['bboxes'])):
            x1, y1, x2, y2 = converted_result['bboxes'][i]
            coord = f"({x1}, {y1}, {x2}, {y2}) "
            bbox_coordinates += coord
        bbox_coordinates = bbox_coordinates.strip()

        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
        """
        The `sv.Detections.from_inference()` method is designed to handle outputs from models that follow a conventional detection format (bounding boxes, scores, and class labels in a standard structure).
        The Florence2 model outputs results in various formats depending on the task and thus requires a specialized method to convert these outputs into a Detections object.
        from_lmm() function is used to specify the dedicated methods for certain models.
        """
        # 이미지, 레이블, coordinate
        
    
    elif mode == IMAGE_CAPTION_GROUNDING_MODE:
        task = FLORENCE_DETAILED_CAPTION_TASK
        _, result = run_florence_inference(
            FLORENCE_MODEL, 
            FLORENCE_PROCESSOR, 
            DEVICE, 
            image_input, 
            task)
        
        text_output = result[task] # 여기서 text_output은 caption

        task = FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
        _, result = run_florence_inference(
            FLORENCE_MODEL, 
            FLORENCE_PROCESSOR, 
            DEVICE, 
            image_input, 
            task,
            text_output)
        
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )

        bbox_coordinates = ""
        for i in range(len(result[task]['bboxes'])):
            x1, y1, x2, y2 = result[task]['bboxes'][i]
            coord = f"({x1}, {y1}, {x2}, {y2}) "
            bbox_coordinates += coord
        bbox_coordinates = bbox_coordinates.strip()
        

         # 이미지, 캡션, coordinate

    return annotate_image(image_input, detections), text_output, bbox_coordinates # annotate result on the original image

@torch.inference_mode()
@torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
def process_stream(frame, conf_threshold=0.3):
    """
    This function is used to process the video frame.
    It takes a frame as input and returns a processed frame.
    """
    frame = cv2.resize(frame, (YOLO_MODEL.input_width, YOLO_MODEL.input_height)) # YOLO 모델에 맞게 크기 조정
    result_frame = YOLO_MODEL.detect_objects(frame, conf_threshold)
    return result_frame

# TODO: WebRTC 화질 조정하기
with gr.Blocks() as demo:
    # 이미지 탭
    with gr.Tab("Image"):
        gr.HTML(
            """
            <h1 style="text-align: center"> Florence2 - Demo </h1>
            """)

        image_processing_mode_dropdown_component = gr.Dropdown(
                choices=IMAGE_INFERENCE_MODES, 
                value=IMAGE_INFERENCE_MODES[0],
                label="Mode",
                info="Select a mode to use.",
                interactive=True
            )
        with gr.Row():
            with gr.Column():
                image_processing_image_input_component = gr.Image(type='pil', label='Upload Image')
                image_processing_text_input_component = gr.Textbox(placeholder = 'Enter comma separated text prompts')
                image_processing_clear_button_component = gr.Button(value='Clear', variant = 'secondary')
                image_processing_submit_button_component = gr.Button(value='Submit', variant = 'primary')
            with gr.Column():
                image_processing_image_output_component = gr.Image(type='pil', label = 'Image Output')
                image_processing_text_output_component = gr.Textbox(label = 'Label Output', visible=True)
                image_processing_boundingbox_coordinates_output_component = gr.Textbox(label = 'Bounding Box Coordinates', visible=True)


        gr.on(
            triggers=[image_processing_submit_button_component.click, image_processing_text_input_component.submit],
            fn = process_image,
            inputs=[
                image_processing_image_input_component,
                image_processing_mode_dropdown_component,
                image_processing_text_input_component],
            outputs=[
                image_processing_image_output_component, 
                image_processing_text_output_component, 
                image_processing_boundingbox_coordinates_output_component
                ]
        )

        image_processing_clear_button_component.click(
            fn=clear_image_processing_components, 
            outputs=[image_processing_image_input_component, 
                    image_processing_text_input_component,
                    image_processing_image_output_component,
                    image_processing_text_output_component,
                    image_processing_boundingbox_coordinates_output_component]
        )
        image_processing_mode_dropdown_component.change(
            fn=on_mode_dropdown_change,
            inputs=image_processing_mode_dropdown_component,
            outputs=[image_processing_text_input_component, image_processing_text_output_component]
        )
    # 웹캠 탭
    with gr.Tab("Video"):
        gr.HTML(
            """
            <h1 style="text-align: center"> YOLOv10 - Demo </h1>
            """)
        stream = WebRTC(label="Webcam", rtc_configuration=None)

    #TODO: pass_frame()을 inference 함수로 대체
    stream.stream(fn=process_stream, inputs=stream, outputs=stream)

# In python, every module has a built-in variable called __name__. When you run a Python file directly, python sets __name__ to "__main__"
# This ensures demo.launch() is only called when you intent run this file directly
if __name__ == "__main__":
    demo.launch(share=True)


