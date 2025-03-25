import autorootcwd
import numpy as np
from ultralytics import YOLO
import cv2
import time

def main():
    model = YOLO('./checkpoints/yolo_v8_tomato.pt')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Unable to open camera. Please check the correct camera index or connection status.")
        return

    window_name = "YOLOv8 Tomato Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Unable to read frame.")
                break

            results = model(frame)
            print(f"results: {results}")

            tomato_boxes = []
            detected_frame = frame.copy()
            for result in results:
                for box in result.boxes:
                    if box.conf >= 0.5: # confidence threshold 

                        xyxy = box.xyxy[0].cpu().numpy().astype(int) # Extract bounding box coordinates (xyxy format) and convert to numpy array
                        tomato_boxes.append(xyxy) # Save tomato bounding box coordinates

                        class_id = int(box.cls[0]) # Get class ID
                        class_name = results[0].names[class_id] # Get class name
                        label = f'{class_name} {box.conf[0]:.2f}' # Create label (class name + confidence)

                        cv2.rectangle(detected_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                        cv2.putText(detected_frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print tomato bounding box coordinates (terminal)
            if tomato_boxes:
                for box in tomato_boxes:
                    x1, y1, x2, y2 = box
                    print(f"Tomato detected! Bounding box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Display detected result image (only tomato)
            cv2.imshow(window_name, detected_frame)

            # Wait for 10 seconds (calculate remaining time)
            elapsed_time = time.time() - start_time
            wait_time = max(1, int(100 - elapsed_time))  # Minimum 0.1 second wait (100ms)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'): # Press 'q' key to exit
                break

    except KeyboardInterrupt:
        print("Program terminated.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()