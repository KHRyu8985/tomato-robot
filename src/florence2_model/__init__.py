import autorootcwd
import torch
import supervision as sv
from .florence import (
    load_florence_model,
    run_florence_inference,
    FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
    FLORENCE_DETAILED_CAPTION_TASK,
    FLORENCE_CAPTION_TASK,
    FLORENCE_MORE_DETAILED_CAPTION_TASK,
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK,
)
from typing import Tuple, Dict, Optional, Union, Any

FLORENCE_MODEL, FLORENCE_PROCESSOR = None, None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700", "#32CD32", "#8A2BE2"]
COLORS = [
    "#FFB6C1",  # Light Pink
    "#B0E0E6",  # Powder Blue
    "#FFDAB9",  # Peach Puff
    "#FFFFE0",  # Light Yellow
    "#98FB98",  # Pale Green
    "#E6E6FA",  # Lavender
]
COLOR_PALETTE = sv.ColorPalette.from_hex(
    COLORS
)  # If no argument is provided, it uses the default palette. (sv.ColorPalette())
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.TOP_LEFT,
    text_color=sv.Color.from_hex("#000000"),
)


def setup_florence_model(device: str = DEVICE):

    # Manually enters a context manager that automatically casts CUDA operations to use bfloat16 precision. (Using bfloat16 can reduce memory usage and improve computation speed without the full loss of accuracy you might get from lower-precesion computations.
    # This is similar to the common use of FP16 autocast but tailored for bfloat16, which is particularly useful on hardware that supports it.)
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    # Enabling TF32 on Supported GPUs
    if (
        torch.cuda.get_device_properties(0).major >= 8
    ):  # This checks the GPU's compute capability. A major version of the 8 or higher indicates an Nvidia Ampere (or later) GPU.
        torch.backends.cuda.matmul.allow_tf32 = True  # If the condition is met, it enables TensorFloat-32 (TF32) support for both CUDA's matrix multiplication (matmul)
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

    global FLORENCE_MODEL, FLORENCE_PROCESSOR

    if FLORENCE_MODEL is None or FLORENCE_PROCESSOR is None:
        FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device)


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
    bboxes = data.get(
        "bboxes", []
    )  # data.get(key, default) -> returns value associated with key if it exsists, otherwise returns default
    labels = data.get("bboxes_labels", [])

    od_results = {"bboxes": bboxes, "labels": labels}
    return od_results


def run_open_vocabulary_detection(image_input, text_input):
    task = FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
    _, result = run_florence_inference(
        FLORENCE_MODEL,
        FLORENCE_PROCESSOR,
        DEVICE,
        image_input,
        task,
        text_input,
    )
    converted_result = convert_to_od_format(result[task])  # convert to OD
    labels = ",".join(converted_result["labels"])

    # change bounding box dictionary to a list
    bbox_coordinates = []
    for i in range(len(converted_result["bboxes"])):
        x1, y1, x2, y2 = converted_result["bboxes"][i]
        coord = [x1, y1, x2, y2]
        bbox_coordinates.append(coord)

    detections = sv.Detections.from_lmm(
        lmm=sv.LMM.FLORENCE_2, result=result, resolution_wh=image_input.size
    )
    """
        The `sv.Detections.from_inference()` method is designed to handle outputs from models that follow a conventional detection format (bounding boxes, scores, and class labels in a standard structure).
        The Florence2 model outputs results in various formats depending on the task and thus requires a specialized method to convert these outputs into a Detections object.
        from_lmm() function is used to specify the dedicated methods for certain models.
    """
    annotated_image = annotate_image(image_input, detections)

    return annotated_image, labels, bbox_coordinates


def run_caption_phrase_grounding(image_input, text_input):
    task = FLORENCE_DETAILED_CAPTION_TASK
    _, result = run_florence_inference(
        FLORENCE_MODEL,
        FLORENCE_PROCESSOR,
        DEVICE,
        image_input,
        task,
    )
    # TODO:text input을 phrase grounding에 반영할 수 있는지 확인

    caption = result[task]

    task = FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
    _, result = run_florence_inference(
        FLORENCE_MODEL,
        FLORENCE_PROCESSOR,
        DEVICE,
        image_input,
        task,
        caption,
    )

    detections = sv.Detections.from_lmm(
        lmm=sv.LMM.FLORENCE_2, result=result, resolution_wh=image_input.size
    )
    bbox_coordinates = []
    for i in range(len(result[task]["bboxes"])):
        x1, y1, x2, y2 = result[task]["bboxes"][i]
        coord = [x1, y1, x2, y2]
        bbox_coordinates.append(coord)

    annotated_image = annotate_image(image_input, detections)

    return annotated_image, caption, bbox_coordinates
