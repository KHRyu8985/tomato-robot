def run_once_on_open():
    print("[INFO] Gesture is changed to open")

def run_once_on_close():
    print("[INFO] Gesture is changed to close")

def run_once_on_pointer():
    print("[INFO] Gesture is changed to pointer")

def run_once_on_stop(landmark_list):
    print("[INFO] Gesture is changed to stop")

def action_for_sign(current_hand_gesture, current_finger_gesture, prev_hand_gesture, prev_finger_gesture, landmark_list):
    point_coords = None

    if current_hand_gesture.lower() == "open" and prev_hand_gesture != "open":
        run_once_on_open()
        point_coords = landmark_list[9]
        prev_hand_gesture = "open"
    if current_hand_gesture.lower() == "close" and prev_hand_gesture != "close":
        run_once_on_close()
        prev_hand_gesture = "close"
    if current_hand_gesture.lower() == "pointer":
        if prev_hand_gesture != "pointer":
            run_once_on_pointer()
            p7 = landmark_list[7]
            p8 = landmark_list[8]

            vec = [p8[0] - p7[0], p8[1] - p7[1]]
            norm = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
            offset = 30

            if norm != 0:
                point_coords = [int(p8[0] + offset * vec[0] / norm), int(p8[1] + offset * vec[1] / norm)]
            else:
                point_coords = [p8[0] + offset, p8[1] + offset]
            prev_hand_gesture = "pointer"
            
        if current_finger_gesture.lower() == "stop":
            if prev_finger_gesture != "stop":
                run_once_on_stop(landmark_list)
            prev_finger_gesture = "stop"

        elif current_finger_gesture.lower() != "stop":
            prev_finger_gesture = "None"
    return prev_hand_gesture, prev_finger_gesture, point_coords

def terminate_for_sign(current_hand_gesture, current_finger_gesture, landmark_list, alert_gesture):
    terminate = False
    point_coords = None

    if current_hand_gesture.lower() == alert_gesture == "pointer":
        terminate = True
        point_coords = landmark_list[8]
    elif current_hand_gesture.lower() == alert_gesture == "open":
        terminate = True
        point_coords = landmark_list[9]

    return terminate, point_coords