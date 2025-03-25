import autorootcwd
import torch
import os
from src.sam2_model.build_sam import build_sam2_camera_predictor

def export_sam2_to_onnx(model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", 
                        checkpoint_path="./checkpoints/sam2.1_hiera_large.pt",
                        output_path="./src/sam2_model/exported_models"):
    
    # set device
    device = torch.device("cpu")  # use CPU for ONNX conversion
    
    # load model
    model = build_sam2_camera_predictor(model_cfg, checkpoint_path)
    model.to(device)
    model.eval()
    
    # create directory
    os.makedirs(output_path, exist_ok=True)
    
    # 1. image encoder conversion
    print("Image encoder conversion in progress...")
    # Image encoder conversion - fixed size used
    dummy_image = torch.randn(1, 3, 1024, 1024, device=device)  # fixed size
    torch.onnx.export(
        model.image_encoder,
        dummy_image,
        f"{output_path}/sam2_image_encoder.onnx",
        opset_version=17,
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamic_axes=None  # disable dynamic axes
    ) 
    
    try:
        torch.onnx.export(
            model.image_encoder,
            dummy_image,
            f"{output_path}/sam2_image_encoder.onnx",
            opset_version=17,
            input_names=["image"],
            output_names=["image_embeddings"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "image_embeddings": {0: "batch", 2: "height", 3: "width"}
            }
        )
        print("Completed image encoder conversion!")
    except Exception as e:
        print(f"Error occurred during image encoder conversion: {e}")
    
    # 2. prompt encoder conversion
    print("Prompt encoder conversion in progress...")
    dummy_point_coords = torch.randint(0, 1024, (1, 1, 2), device="cpu", dtype=torch.float32)
    dummy_point_labels = torch.ones((1, 1), dtype=torch.int64, device="cpu")  # int64로 명시
    
    class PromptEncoderWrapper(torch.nn.Module):
        def __init__(self, prompt_encoder):
            super().__init__()
            self.prompt_encoder = prompt_encoder
            
        def forward(self, point_coords, point_labels):
            return self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None
            )
    
    prompt_encoder_wrapper = PromptEncoderWrapper(model.sam_prompt_encoder)
    
    try:
        torch.onnx.export(
            prompt_encoder_wrapper,
            (dummy_point_coords, dummy_point_labels),
            f"{output_path}/sam2_prompt_encoder.onnx",
            opset_version=17,
            input_names=["point_coords", "point_labels"],
            output_names=["sparse_embeddings", "dense_embeddings"],
            dynamic_axes={
                "point_coords": {0: "batch", 1: "num_points"},
                "point_labels": {0: "batch", 1: "num_points"}
            }
        )
        print("Completed prompt encoder conversion!")
    except Exception as e:
        print(f"Error occurred during prompt encoder conversion: {e}")
    
    print(f"Converted model is saved in {output_path}")

if __name__ == "__main__":
    export_sam2_to_onnx()
