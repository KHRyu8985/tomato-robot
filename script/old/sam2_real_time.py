import autorootcwd
import sys
import torch
import cv2
import numpy as np

from src.sam2_model.build_sam import build_sam2_camera_predictor
from src.sam2_model.utils.prompt import mouse_callback_real_time, process_keyboard_input
from src.sam2_model.utils.real_time import show_mask_overlay, show_prompt_on_frame

def sam2_real_time_inference(mode, point_coords = None, point_labels = None, bbox = None):
    # initialize model
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
    
    # initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera. Please check the correct camera index or connection status.")
        sys.exit(1)

    # initialize prompt
    prompt = {
        "mode": mode,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "bbox_start": None,
        "bbox_end": None,
        "bbox": bbox,
        "if_init": False
    }
    
    # initialize window
    cv2.namedWindow("Real-Time Camera")
    cv2.setMouseCallback("Real-Time Camera", mouse_callback_real_time, prompt)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_obj_ids, out_mask_logits = None, None
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1) #Mirror display
            
            if not ret:
                break
            # get key input and update prompt values via function call
            if process_keyboard_input(prompt):
                break
            # if new prompt is input, initialize tracker
            if not prompt["if_init"]:
                if prompt["mode"] == "point" and prompt["point_coords"]:
                    predictor.load_first_frame(frame)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        0, 1, points=prompt["point_coords"], labels=prompt["point_labels"]
                    )
                    prompt["if_init"] = True
                elif prompt["mode"] == "box" and prompt["bbox"] is not None:
                    predictor.load_first_frame(frame)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        0, 1, bbox=prompt["bbox"]
                    )
                    prompt["if_init"] = True
            else:
                out_obj_ids, out_mask_logits = predictor.track(frame)
            
            show_prompt_on_frame(frame, prompt) # draw prompt information on the original frame

            if out_mask_logits is not None:
                overlay, _ = show_mask_overlay(frame, out_mask_logits, prompt) # show mask overlay (show center information of the object)
                cv2.imshow("Mask Prediction", overlay)
        
    cap.release()
    cv2.destroyAllWindows()

def main():
    sam2_real_time_inference(mode="point", point_coords=[[500, 500]], point_labels=[1])
   

if __name__ == "__main__":
    main()