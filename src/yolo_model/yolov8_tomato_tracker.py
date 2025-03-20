import cv2
from ultralytics import YOLO

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

    def detect_tomatoes(self, frame, confidence_threshold=0.5):
        """
        Detect tomatoes in the frame using YOLOv8 model and return bounding box information.
        """

        if self.model is None:
            print("[ERROR] YOLOv8 model is not loaded.")
            return frame, [], None

        results = self.model(frame, verbose=False)
        tomato_boxes = []
        detected_frame = frame.copy()

        for result in results:
            for box in result.boxes:
                if box.conf >= confidence_threshold:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    tomato_boxes.append(xyxy)

                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    label = f'{class_name} {box.conf[0]:.2f}'

                    cv2.rectangle(detected_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    cv2.putText(detected_frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detected_frame, tomato_boxes, results