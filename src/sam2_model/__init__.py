# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import autorootcwd
import torch
import cv2
import numpy as np
from src.sam2_model.build_sam import build_sam2_camera_predictor

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("src.sam2_model", version_base="1.2")

SAM2_PREDICTOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_sam2_model(checkpoint, config, device=DEVICE):
    global SAM2_PREDICTOR
    print(f"Using device: {device}")
    if SAM2_PREDICTOR is None:
        SAM2_PREDICTOR = build_sam2_camera_predictor(config, checkpoint, device=device)
    # return SAM2_PREDICTOR


def process_new_prompt(frame, points, labels):
    global SAM2_PREDICTOR
    SAM2_PREDICTOR.load_first_frame(frame)
    _, out_obj_ids, out_mask_logits = SAM2_PREDICTOR.add_new_prompt(
        frame_idx=0, obj_id=1, points=points, labels=labels
    )
    overlay = draw_mask(frame, out_mask_logits)
    return overlay

def process_new_box(frame, bbox):
    global SAM2_PREDICTOR
    SAM2_PREDICTOR.load_first_frame(frame)
    _, out_obj_ids, out_mask_logits = SAM2_PREDICTOR.add_new_prompt(
        frame_idx=0, obj_id=1, bbox=bbox
    )
    overlay = draw_mask(frame, out_mask_logits)
    return overlay

def track_object(frame):
    global SAM2_PREDICTOR
    out_obj_ids, out_mask_logits = SAM2_PREDICTOR.track(frame)
    overlay = draw_mask(frame, out_mask_logits)
    return overlay


def draw_mask(frame, mask):
    if mask is None:
        return frame

    if isinstance(mask, torch.Tensor):
        mask_pred = mask.detach().cpu().numpy()
        """
            PyTorch tensors can reside on different devices - either on the CPU or the GPU.
            When a tensor is on the GPU(i.e., a CUDA tensor), its data is stored in GPU memory.
            However, NumPy arrays are stored in CPU memory and cannot directly access GPU memory.
            Thus, a transferring step to the CPU is required.
            """
    mask_pred = np.squeeze(mask_pred)  # Extra dimensions are removed
    binary_mask = (mask_pred > 0.5).astype(np.uint8) * 255  # 흰색으로 변환
    # Mask values are thresholded at 0.5 (values above 0.5 are set to True, others to False)
    # The boolean array is cast to an 8-bit unsigned integer and multiplied by 255 (Foreground: white(255), Background: black(0))
    if binary_mask.ndim != 2:  # if mask dimension is not 2
        # reset shape to match the last two dimensions (usually the height and width of the frame)
        binary_mask = binary_mask.reshape(binary_mask.shape[-2], binary_mask.shape[-1])
        # binary mask is converted into a colored image using OpenCV's applyColorMap function with the COLORMAP_JET option
    
    # Create an overlay effect with transparency
    alpha = 0.85
    mask_bool = (binary_mask == 255)
    mask_bool_3c = np.repeat(mask_bool[:, :, None], 3, axis=2)  # Convert to 3 channels
    dark_frame = (frame * (1.0 - alpha)).astype(np.uint8)  # Darkened version of the frame
    overlay = np.where(mask_bool_3c, frame, dark_frame)  # Keep masked region, darken background

    # colored_mask = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
    # overlay = cv2.addWeighted(
    #     frame, 0.5, colored_mask, 0.7, 0
    # )  # the original frame and the colored mask are blened together with 30% of the original frame and

    return overlay
