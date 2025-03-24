import autorootcwd
import cv2
import numpy as np
import time

from src.yolo_model.yolov8_tomato_tracker import YOLOv8TomatoTracker
from src.sam2_model.sam2_tracker import SAM2Tracker

def main():
    yolo_tracker = YOLOv8TomatoTracker()
    sam2_tomato_tracker = SAM2Tracker(class_name="tomato")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera.")
        return

    cv2.namedWindow("SAM Segmentation with YOLO Tomato Detection", cv2.WINDOW_NORMAL)

    yolo_interval = 0.1 # YOLO detection interval (seconds)
    last_yolo_time = time.time() - yolo_interval

    sam2_interval = 3.0  # SAM2 segmentation interval (3 seconds)
    last_sam2_time = time.time() - sam2_interval

    tomato_boxes_buffer = []
    mask_image = None

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Unable to read frame.")
                break

            current_time = time.time()
            if current_time - last_yolo_time >= yolo_interval:
                detected_frame_yolo, tomato_boxes, yolo_results = yolo_tracker.detect_tomatoes(frame)
                tomato_boxes_buffer = tomato_boxes
                last_yolo_time = current_time

            if current_time - last_sam2_time >= sam2_interval:
                if tomato_boxes_buffer:
                    print("SAM2 segmentation performed")
                    tomato_detection, sam2_mask_image = sam2_tomato_tracker.get_tomato_mask(frame, tomato_boxes_buffer)
                    last_sam2_time = current_time

            cv2.imshow("YOLO Tomato Detection", detected_frame_yolo)
            if sam2_mask_image is not None:
                cv2.imshow("SAM Segmentation with YOLO Tomato Detection", sam2_mask_image)

            elapsed_time = time.time() - start_time
            wait_time = max(1, int(100 - elapsed_time))
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program terminated.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
