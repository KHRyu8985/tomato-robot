import torch
import cv2
import numpy as np

from src.sam2_model.build_sam import build_sam2_camera_predictor
from src.sam2_model.utils.real_time import show_mask_overlay
from src.sam2_model.utils.feature_extraction import apply_pca_and_visualize

class SAM2Tracker:
    def __init__(self, checkpoint_path="./checkpoints/sam2.1_hiera_large.pt", model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
        self.predictor = build_sam2_camera_predictor(model_cfg, checkpoint_path)
        self.prompt = {"mode": "point", "point_coords": None, "point_labels": [1], "bbox": None, "bbox_start": None, "bbox_end": None, "if_init": False}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process_frame(self, image, debug_image, point_coords):
        if point_coords is not None:
            self.prompt["point_coords"] = [point_coords]
            self.prompt["point_labels"] = [1]
            self.prompt["if_init"] = False

        out_mask_logits = None

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self.prompt["point_coords"] is not None:
                if not self.prompt["if_init"]:
                    self.predictor.load_first_frame(image)
                    _, _, _ = self.predictor.add_new_prompt(0, 1, points=self.prompt["point_coords"], labels=self.prompt["point_labels"])
                    self.prompt["if_init"] = True
                else:
                    _, out_mask_logits = self.predictor.track(image)

        if out_mask_logits is not None:
            debug_image = show_mask_overlay(debug_image, out_mask_logits, self.prompt)

        return debug_image

    def process_frame_with_visualization(self, image, debug_image, point_coords):
        if point_coords is not None:
            self.prompt["point_coords"] = [point_coords]
            self.prompt["point_labels"] = [1]
            self.prompt["if_init"] = False

        out_mask_logits = None
        pca_visualization = None

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self.prompt["point_coords"] is not None:
                if not self.prompt["if_init"]:
                    self.predictor.load_first_frame(image)

                    features = {}
                    def hook_fn(name):
                        def hook(module, input, output):
                            features[name] = output
                        return hook
                    hooks = []

                    mask_decoder_model = self.predictor.sam_mask_decoder.to(self.device)
                    hooks.append(mask_decoder_model.transformer.layers[1].register_forward_hook(hook_fn("layer2")))

                    _, _, _ = self.predictor.add_new_prompt(0, 1, points=self.prompt["point_coords"], labels=self.prompt["point_labels"])
                    self.prompt["if_init"] = True

                    pca_features = {}
                    for name, feature in features.items():
                        for idx, item in enumerate(feature):
                            for jdx, tensor in enumerate(item):
                                print(f"{name}_{idx}_{jdx} shape: {tensor.shape}")
                                if idx == 1:
                                    pca_img = apply_pca_and_visualize("stream", tensor, image, "mask_decoder", save_path = None, visualize = False)
                                    pca_img = (np.clip(pca_img, 0, 1) * 255).astype(np.uint8)

                                    height, width = image.shape[:2]
                                    target_size = (width // 2, height // 2)
                                    pca_img_resized = cv2.resize(pca_img, target_size)
                                    
                                    pca_visualization = pca_img_resized
                                    # cv2.imshow("PCA Visualization", pca_visualization)
                    for hook in hooks:
                        hook.remove()
                else:
                    _, out_mask_logits = self.predictor.track(image)

        if out_mask_logits is not None:
            debug_image = show_mask_overlay(debug_image, out_mask_logits, self.prompt)

        if pca_visualization is not None:
            return debug_image, pca_visualization
        else:
            return debug_image, None
