import autorootcwd
import click
import cv2 as cv
import numpy as np
import pyzed.sl as sl

from src.hand_gesture.hand_tracker import HandTracker
from src.sam2_model.sam2_tracker import SAM2Tracker
from src.zed_sdk.zed_tracker import ZedTracker

@click.command()
@click.option('--camera', type=str, default='zed', help='camera type (zed, femto)')
@click.option('--show-feature', is_flag=True, default=False, help='visualization: mask decoder feature extraction result')
def pipeline(camera='zed', show_feature=False):
    if camera == 'zed':
        zed_tracker = ZedTracker()
        if not zed_tracker.initialize_zed():
            return
    else:
        cap = cv.VideoCapture(0)
        zed_tracker = None

    hand_tracker = HandTracker()
    sam2_tracker = SAM2Tracker()
    
    while True:
        if camera == 'zed':
            success, frame, objects = zed_tracker.grab_frame_and_objects()
            if not success:
                continue

            zed_tracker.update_viewer()

            debug_image = frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("[Error] failed to read frame")
                break

        debug_image = frame.copy()

        debug_image, point_coords = hand_tracker.process_frame(frame, debug_image, None, None)
        
        if show_feature:
            debug_image, pca_visualization = sam2_tracker.process_frame_with_visualization(frame, debug_image, point_coords)
            cv.imshow('hand gesture recognition', debug_image)
            if pca_visualization is not None:
                cv.imshow('PCA visualization', pca_visualization)
        else:
            debug_image, _ = sam2_tracker.process_frame(frame, debug_image, point_coords)
            cv.imshow('hand gesture recognition', debug_image)

        if cv.waitKey(10) & 0xFF == 27:
            break

    if camera == 'zed':
        zed_tracker.close_zed()
    else:
        cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    pipeline()