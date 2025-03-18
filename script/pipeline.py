import autorootcwd
import click
import cv2 as cv
import numpy as np

from src.hand_gesture.hand_tracker import HandTracker
from src.sam2_model.sam2_tracker import SAM2Tracker

@click.command()
@click.option('--device', type=int, default=0, help='Camera device index')
@click.option('--show-feature', is_flag=True, default=False, help='If set, visualize mask decoder feature extraction results.')
def pipeline(device=0, show_feature=False):
    cap = cv.VideoCapture(device)
    hand_tracker = HandTracker()
    sam2_tracker = SAM2Tracker()
    
    while True:
        ret, image = cap.read()
        if not ret:
            print("[Error] Failed to read frame")
            break
        debug_image = image.copy()

        debug_image, point_coords = hand_tracker.process_frame(image, debug_image, None, None)
        if show_feature:
            debug_image, pca_visualization = sam2_tracker.process_frame_with_visualization(image, debug_image, point_coords)
            cv.imshow('Hand Gesture Recognition', debug_image)
            if pca_visualization is not None:
                cv.imshow('PCA Visualization', pca_visualization)
        else:
            debug_image, _ = sam2_tracker.process_frame(image, debug_image, point_coords)
            cv.imshow('Hand Gesture Recognition', debug_image)

        if cv.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    pipeline()
