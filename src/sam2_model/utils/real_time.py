import cv2
import numpy as np
import torch

def show_prompt_on_frame(frame, prompt):
    """
    Show the prompt information on the frame.
    
    Args:
        frame (numpy.ndarray): original image
        prompt (dict): dictionary containing prompt information ('point' or 'box' mode)

    """
    frame_display = frame.copy()
    
    if prompt["mode"] == "point":
        for pt in prompt["point_coords"]:
            cv2.circle(frame_display, (int(pt[0]), int(pt[1])), radius=5, color=(0, 255, 0), thickness=-1)
    elif prompt["mode"] == "box":
        if prompt["bbox"] is not None:
            x_min, y_min, x_max, y_max = prompt["bbox"][0]
            cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
        elif prompt["bbox_start"] is not None and prompt["bbox_end"] is not None:
            start = prompt["bbox_start"]
            end = prompt["bbox_end"]
            x_min = int(min(start[0], end[0]))
            y_min = int(min(start[1], end[1]))
            x_max = int(max(start[0], end[0]))
            y_max = int(max(start[1], end[1]))
            cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    
    cv2.imshow("Real-Time Camera", frame_display)

def show_mask_overlay(frame, out_mask_logits, prompt):
    """
    Build a mask overlay image using the given frame, mask logits, and prompt information.
    
    Args:
        frame (numpy.ndarray): current frame
        out_mask_logits (torch.Tensor or numpy.ndarray): model's mask logits
        prompt (dict): prompt related information (e.g. mode, point_coords, bbox, etc.)
        
    """
    # convert torch tensor to numpy array if it is a torch tensor
    mask_pred = out_mask_logits.cpu().numpy() if isinstance(out_mask_logits, torch.Tensor) else out_mask_logits
    mask_pred = np.squeeze(mask_pred)
    binary_mask = (mask_pred > 0.5).astype(np.uint8) * 255
    if binary_mask.ndim != 2:
        binary_mask = binary_mask.reshape(binary_mask.shape[-2], binary_mask.shape[-1])
    
    # create overlay
    alpha = 0.85
    mask_bool = (binary_mask == 255)
    mask_bool_3c = np.repeat(mask_bool[:, :, None], 3, axis=2)
    dark_frame = (frame * (1.0 - alpha)).astype(np.uint8)
    overlay = np.where(mask_bool_3c, frame, dark_frame)
    
    # draw prompt information
    if prompt["mode"] == "point":
        for pt in prompt["point_coords"]:
            x, y = int(pt[0]), int(pt[1])
            size = 7
            cv2.line(overlay, (x - size, y), (x + size, y), color=(255, 255, 255), thickness=3)
            cv2.line(overlay, (x, y - size), (x, y + size), color=(255, 255, 255), thickness=3)
            cv2.line(overlay, (x - size, y), (x + size, y), color=(0, 0, 0), thickness=1)
            cv2.line(overlay, (x, y - size), (x, y + size), color=(0, 0, 0), thickness=1)
    elif prompt["mode"] == "box" and prompt["bbox"] is not None:
        x_min, y_min, x_max, y_max = prompt["bbox"][0]
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    
    # calculate center coordinates (use moments)
    moments = cv2.moments(binary_mask)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        cv2.circle(overlay, (cX, cY), radius=5, color=(255, 255, 255), thickness=-1)
        cv2.putText(
            overlay,
            f"({cX}, {cY})",
            (cX - 20, cY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    
    # cv2.imshow("Predicted Mask", overlay)
    return overlay, binary_mask