import autorootcwd
import cv2
from ultralytics import YOLO
import numpy as np

from src.yolo_model.utils import compute_iou

class YOLOv8TomatoTracker:
    def __init__(self, model_path='./checkpoints/yolo_v8_tomato.pt'):
        self.model = self.load_yolov8_model(model_path)
        if self.model is None:
            raise Exception("Failed to initialize YOLOv8 model")

    def load_yolov8_model(self, model_path):
        """Load YOLOv8 model."""
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOv8 model: {e}")
            return None
        
    def nms_tomato_boxes(self, tomato_boxes, tomato_confs, iou_threshold=0.5):
        """
        Perform non-maximum suppression on tomato bounding boxes.
        """
        # Remove duplicate bounding boxes
        nms_tomato_boxes = []

        if tomato_boxes:
            boxes_array = np.array(tomato_boxes)
            scores_array = np.array(tomato_confs)
            
            sorted_indices = np.argsort(scores_array)[::-1]
            keep_indices = []
            
            # NMS
            while len(sorted_indices) > 0:
                current_index = sorted_indices[0]
                keep_indices.append(current_index)
                
                if len(sorted_indices) == 1:
                    break
                
                current_box = boxes_array[current_index]
                rest_boxes = boxes_array[sorted_indices[1:]]
                
                ious = compute_iou(current_box, rest_boxes)
                
                mask = ious < iou_threshold
                sorted_indices = sorted_indices[1:][mask]

            nms_tomato_boxes = [tomato_boxes[i] for i in keep_indices]
     
        return nms_tomato_boxes
            

    def detect_tomatoes(self, frame, confidence_threshold=0.5):
        """
        Detect tomatoes in the frame using YOLOv8 model and return bounding box information.
        """

        if self.model is None:
            print("[ERROR] YOLOv8 model is not loaded.")
            return frame, [], None

        results = self.model(frame, verbose=False)
        tomato_boxes = [] # bounding box coordinates
        tomato_confs = [] # confidence score
        detected_frame = frame.copy()

        for result in results:
            for box in result.boxes:
                if box.conf >= confidence_threshold:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    tomato_boxes.append(xyxy)
                    tomato_confs.append(float(box.conf))

                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    label = f'{class_name} {box.conf[0]:.2f}'

                    cv2.rectangle(detected_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    cv2.putText(detected_frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # NMS
        tomato_boxes = self.nms_tomato_boxes(tomato_boxes, tomato_confs, iou_threshold=0.5)
        
        if tomato_boxes:
            tomato_boxes.sort(key=lambda box: (box[0], box[1])) 

        return detected_frame, tomato_boxes, results