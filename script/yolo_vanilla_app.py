import av 
import cv2
import tempfile
import supervision as sv
import gradio as gr
from gradio_webrtc import WebRTC
from huggingface_hub import hf_hub_download
from path_setup import setup_project_root
setup_project_root()
from src.yolo_model.yolov10 import YOLOv10



model_file = hf_hub_download(
    repo_id="onnx-community/yolov10n", filename="onnx/model.onnx"
)

model = YOLOv10(model_file)


rtc_configuration = None


# def pass_frame(frame):
#     # The Femto-bolt camera returns frames in a different format, thus requires additional processing
#     # The frame argument is often a av.VideoFrame object from PyAV
#     return frame



def pass_frame(frame: av.VideoFrame) -> av.VideoFrame:
     # Convert the incoming frame to a NumPy array (BGR)
    image_bgr = frame.to_ndarray(format="bgr24")

    # Conver the processed NumPy array back to an av.VideoFrame object
    return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


def detection(image, conf_threshold=0.3):
    image = cv2.resize(image, (model.input_width, model.input_height))
    new_image = model.detect_objects(image, conf_threshold)

    return new_image

   

css = """.my-group {max-width: 600px !important; max-height: 600px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items:center !important;}"""
with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style="text-align: center"> YOLOv10 - Demo </h1>
        """
    )

    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
           
            image = WebRTC(label="Stream", rtc_configuration=rtc_configuration) # WebRTC typically requires a valid RTC configuration dictionary
            image.stream(
                fn=detection, inputs=image, outputs=image
            )



# In python, every module has a built-in variable called __name__. When you run a Python file directly, python sets __name__ to "__main__"
# This ensures demo.launch() is only called when you intent run this file directly
if __name__ == "__main__":
    demo.launch(share=True)