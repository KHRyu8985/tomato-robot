import autorootcwd
import torch
import cv2
import numpy as np
import click

from src.sam2_model.build_sam import build_sam2_image_predictor, build_sam2_camera_predictor
from src.sam2_model.utils.feature_extraction import apply_pca_and_visualize
from src.sam2_model.utils.prompt import get_click_point
from src.sam2_model.utils.real_time import show_prompt_on_frame, show_mask_overlay
from src.sam2_model.utils.prompt import mouse_callback_real_time, process_keyboard_input

@click.command()
@click.option(
    '--mode',
    type=click.Choice(['image', 'video', 'stream']),
    default='stream',
    help='Feature visualization mode (image, video, stream)'
)
def main(mode):

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if mode == 'image':
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        if not image_path:
            raise ValueError("Image file is not selected.")

        # initialize the predictor
        predictor = build_sam2_image_predictor(model_cfg, sam2_checkpoint)
        mask_decoder_model = predictor.model.sam_mask_decoder.to(device).eval()
        
        features = {}
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook

        # mask decoder
        hooks = []
        hooks.append(mask_decoder_model.transformer.layers[1].register_forward_hook(hook_fn("layer2")))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # set point prompt
        input_points, input_labels = [], []
        input_points, input_labels = get_click_point(image, input_points, input_labels)

        # Forward Pass
        with torch.no_grad():
            predictor.set_image(image)
            _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True,
                )

        # apply PCA and save individual plots (save PCA result images to dictionary)
        pca_features = {}
        for name, feature in features.items():
            for idx, item in enumerate(feature):
                for jdx, tensor in enumerate(item):
                    print(f"{name}_{idx}_{jdx} shape: {tensor.shape}")
                    print("image.shape: ", image.shape)
                    idx_name = name+f"_{idx}_{jdx}"
                    save_path = None
                    if idx == 1:
                        pca_features[idx_name] = apply_pca_and_visualize(mode, tensor, image, "mask_decoder", save_path, visualize=True)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    elif mode == 'video':
        pass
    
    elif mode == 'stream':
       # initialize model
        predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        
        # initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open camera. Please check the correct camera index or connection status.")
            return

        # initialize prompt
        prompt = {
            "mode": "point",
            "point_coords": [[500, 500]],
            "point_labels": [1],
            "bbox_start": None,
            "bbox_end": None,
            "bbox": None,
            "if_init": False
        }
        
        # initialize window for stream and PCA visualization
        cv2.namedWindow("Real-Time Camera")
        cv2.namedWindow("PCA Visualization")
        cv2.setMouseCallback("Real-Time Camera", mouse_callback_real_time, prompt)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_obj_ids, out_mask_logits = None, None
            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)  # Mirror display
                
                if not ret:
                    break

                if process_keyboard_input(prompt):
                    break

                if not prompt["if_init"]:
                    if prompt["mode"] == "point" and prompt["point_coords"]:
                        predictor.load_first_frame(frame)
                        
                        features = {}
                        def hook_fn(name):
                            def hook(module, input, output):
                                features[name] = output
                            return hook
                        hooks = []

                        mask_decoder_model = predictor.sam_mask_decoder.to(device)
                        hooks.append(mask_decoder_model.transformer.layers[1].register_forward_hook(hook_fn("layer2")))
                        
                        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                            0, 1, points=prompt["point_coords"], labels=prompt["point_labels"]
                        )
                        prompt["if_init"] = True
                        
                        pca_features = {}
                        for name, feature in features.items():
                            for idx, item in enumerate(feature):
                                for jdx, tensor in enumerate(item):
                                    print(f"{name}_{idx}_{jdx} shape: {tensor.shape}")
                                    if idx == 1:
                                        pca_img = apply_pca_and_visualize(mode, tensor, frame, "mask_decoder", save_path=None, visualize=False)
                                        pca_img = (np.clip(pca_img, 0, 1) * 255).astype(np.uint8)

                                        height, width = frame.shape[:2]
                                        target_size = (width // 2, height // 2)
                                        pca_img_resized = cv2.resize(pca_img, target_size)

                                        cv2.imshow("PCA Visualization", pca_img_resized)
                        for hook in hooks:
                            hook.remove()
                else:
                    out_obj_ids, out_mask_logits = predictor.track(frame)
                
                show_prompt_on_frame(frame, prompt)
                if out_mask_logits is not None:
                    show_mask_overlay(frame, out_mask_logits, prompt)
                cv2.imshow("Real-Time Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()