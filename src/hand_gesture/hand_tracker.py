import autorootcwd
import csv
import copy
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import json
from datetime import datetime
import os
import time

from src.hand_gesture.model import KeyPointClassifier, PointHistoryClassifier
from src.hand_gesture.utils.logging import logging_csv
from src.hand_gesture.utils.preprocessing import pre_process_landmark, pre_process_point_history
from src.hand_gesture.utils.visualization import calc_bounding_rect, calc_landmark_list, draw_bounding_rect, draw_landmarks, draw_info_text, draw_point_history, draw_info
from src.hand_gesture.utils.actions import action_for_sign

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        # Load labels
        self.keypoint_classifier_labels = self._load_labels('src/hand_gesture/model/keypoint_classifier/keypoint_classifier_label.csv')
        self.point_history_classifier_labels = self._load_labels('src/hand_gesture/model/point_history_classifier/point_history_classifier_label.csv')

        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)

        self.prev_hand_gesture = "None"
        self.prev_finger_gesture = "None"

        self.tomato_coords_file = 'src/hand_gesture/data/tomato_coordinates.json'
        self.tomato_coordinates = self.load_tomato_coordinates()

        self.recording_tomato_id = None
        self.selection_start_time = None
        self.last_nearest_tomato = None
        self.selected_tomato = None

    def _load_labels(self, filepath):
        with open(filepath, encoding='utf-8-sig') as f:
            return [row[0] for row in csv.reader(f)]

    def load_tomato_coordinates(self):
        try:
            with open(self.tomato_coords_file, 'r') as f:
                data = f.read().strip()
                if data:
                    return json.loads(data)
                else:
                    return self._create_initial_structure()
        except (FileNotFoundError, json.JSONDecodeError):
            return self._create_initial_structure()

    def _create_initial_structure(self):
        """create the initial data structure"""
        initial_data = {f"tomato_{i}": {"coordinates": [], "center": None} for i in range(1, 5)}
        
        os.makedirs(os.path.dirname(self.tomato_coords_file), exist_ok=True)
        
        with open(self.tomato_coords_file, 'w') as f:
            json.dump(initial_data, f, indent=4)
        
        return initial_data
    
    def save_tomato_coordinate(self, tomato_id, point_coords):
        """save the tomato pointing coordinates"""
        if not isinstance(tomato_id, int) or not (1 <= tomato_id <= 4):
            return False
            
        coord = {
            "x": int(point_coords[0]),
            "y": int(point_coords[1]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        tomato_key = f"tomato_{tomato_id}"
        self.tomato_coordinates[tomato_key]["coordinates"].append(coord)
        self._update_tomato_center(tomato_id)
        
        with open(self.tomato_coords_file, 'w') as f:
            json.dump(self.tomato_coordinates, f, indent=4)
        
        return True
    
    def _update_tomato_center(self, tomato_id):
        """update the average center coordinates of the tomato"""
        tomato_key = f"tomato_{tomato_id}"
        coords = self.tomato_coordinates[tomato_key]["coordinates"]
        
        if coords:
            x_mean = sum(c["x"] for c in coords) / len(coords)
            y_mean = sum(c["y"] for c in coords) / len(coords)
            
            self.tomato_coordinates[tomato_key]["center"] = {
                "x": int(x_mean),
                "y": int(y_mean)
            }
    
    def find_nearest_tomato(self, current_point):
        """find the nearest tomato from current point"""
        min_dist = float('inf')
        nearest_tomato = None
        
        for tomato_id in range(1, 5):
            tomato_key = f"tomato_{tomato_id}"
            center = self.tomato_coordinates[tomato_key]["center"]
            
            if center:
                dist = np.sqrt(
                    (current_point[0] - center["x"])**2 + 
                    (current_point[1] - center["y"])**2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_tomato = tomato_id
        
        return nearest_tomato, min_dist

    def draw_tomato_centers(self, debug_image):
        """draw the tomato centers on the image"""
        for tomato_id in range(1, 5):
            tomato_key = f"tomato_{tomato_id}"
            center = self.tomato_coordinates[tomato_key]["center"]
            
            if center:
                # display the center point in red
                cv.circle(debug_image, (center["x"], center["y"]), 5, (0, 0, 255), -1)
                # display the tomato number
                cv.putText(debug_image, 
                          f"#{tomato_id}", 
                          (center["x"] + 10, center["y"] + 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 
                          0.6, 
                          (0, 0, 255), 
                          2)
        return debug_image

    def process_frame(self, image, debug_image, number, key, use_point_tracker=False, mode=None):
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        prompt_point = None
        expected_point_coords = None
        landmark_list = None

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, self.point_history)

                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])

                point_history_len = len(pre_processed_point_history_list)
                finger_gesture_id = 0
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)

                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                current_hand_gesture = self.keypoint_classifier_labels[hand_sign_id]
                current_finger_gesture = self.point_history_classifier_labels[most_common_fg_id[0][0]]

                self.prev_hand_gesture, self.prev_finger_gesture, point_coords, expected_point_coords = action_for_sign(
                    current_hand_gesture, current_finger_gesture,
                    self.prev_hand_gesture, self.prev_finger_gesture, landmark_list, use_point_tracker
                )

                if point_coords is not None:
                    prompt_point = point_coords

                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

        else:
            self.point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, self.point_history)
        debug_image = self.draw_tomato_centers(debug_image)

        return debug_image, prompt_point, expected_point_coords, landmark_list