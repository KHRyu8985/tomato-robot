import cv2
import numpy as np

def get_click_point(image, input_points, input_labels):
    """
    Show the click point of the object in the image, and update the current prompt (click point) in real-time.
    
    When the left click is made, the green circle and '+' symbol are displayed in the image.
    When the right click is made, the red circle and '-' symbol are displayed in the image.
    """
    points, labels = input_points, input_labels
    img_show = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    def mouse_callback(event, x, y, flags, param):
        nonlocal img_show
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(1)
            cv2.circle(img_show, (x, y), 15, (0, 200, 0), 2)
            cv2.putText(img_show, "+", (x - 13, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 200, 0), 2, cv2.LINE_AA)
            cv2.imshow("Select Object", img_show)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y))
            labels.append(0)
            cv2.circle(img_show, (x, y), 15, (0, 0, 200), 2)
            cv2.putText(img_show, "-", (x - 13, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 200), 2, cv2.LINE_AA)
            cv2.imshow("Select Object", img_show)

    cv2.namedWindow("Select Object")
    cv2.setMouseCallback("Select Object", mouse_callback)
    cv2.imshow("Select Object", img_show)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return np.array(points), np.array(labels)

def get_bounding_box(image):
    """
    Get the bounding box of the object in the image
    """
    points = []
    temp_point = None
    drawing = False
    img_copy = image.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp_point, drawing, img_copy
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if not points:
                drawing = True
                points.append((x, y))
                temp_point = (x, y)
                img_copy = image.copy()
            elif len(points) == 1:
                points.append((x, y))
                cv2.rectangle(img_copy, points[0], points[1], (0, 255, 0), 2)
            
        elif event == cv2.EVENT_MOUSEMOVE and drawing and len(points) == 1:
            img_copy = image.copy()
            cv2.rectangle(img_copy, points[0], (x, y), (0, 255, 0), 2)

    cv2.imshow("Draw Bounding Box", cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback("Draw Bounding Box", mouse_callback)
    
    while len(points) < 2:
        cv2.imshow("Draw Bounding Box", cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return np.array(points)

def mouse_callback_real_time(event, x, y, flags, prompt):
    """
    mouse callback function for real-time prompt input (point or bounding box)

    prompt: dictionary instead of global variables (e.g. popup window state and prompt information)
    """
    prompt_mode = prompt.get("mode", "point")
    
    if prompt_mode == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            prompt["point_coords"] = [[x, y]]
            prompt["point_labels"] = [1]
            prompt["if_init"] = False
            print(f"New point prompt input: {prompt['point_coords']}")
    elif prompt_mode == "box":
        if event == cv2.EVENT_LBUTTONDOWN:
            prompt["bbox"] = None
            prompt["bbox_start"] = (x, y)
            prompt["bbox_end"] = (x, y)
            print(f"Bounding box starting point: {prompt['bbox_start']}")
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if prompt.get("bbox_start") is not None:
                prompt["bbox_end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            prompt["if_init"] = False
            if prompt.get("bbox_start") is not None:
                prompt["bbox_end"] = (x, y)
                x_min = min(prompt["bbox_start"][0], prompt["bbox_end"][0])
                y_min = min(prompt["bbox_start"][1], prompt["bbox_end"][1])
                x_max = max(prompt["bbox_start"][0], prompt["bbox_end"][0])
                y_max = max(prompt["bbox_start"][1], prompt["bbox_end"][1])
                prompt["bbox"] = [[x_min, y_min, x_max, y_max]]
                print(f"New bounding box prompt input: {prompt['bbox']}")

def process_keyboard_input(prompt):
    """
    Process keyboard input to update prompt mode
    
    Input:
        prompt(dict): dictionary containing prompt settings
    Returns:
        True: user requests termination by pressing 'q' key
        False: other cases
    """
    key = cv2.waitKey(1) & 0xFF
    if key == ord("p"): # convert to point mode
        prompt["mode"] = "point"
        prompt["point_coords"] = []
        prompt["point_labels"] = []
        prompt["bbox_start"] = None
        prompt["bbox_end"] = None
        prompt["bbox"] = None
        prompt["if_init"] = False
        print("Point prompt mode activated.")
    elif key == ord("b"): # convert to bounding box mode
        prompt["mode"] = "box"
        prompt["point_coords"] = []
        prompt["point_labels"] = []
        prompt["bbox_start"] = None
        prompt["bbox_end"] = None
        prompt["bbox"] = None
        prompt["if_init"] = False
        print("Bounding box prompt mode activated.")
    elif key == ord("q"): # quit
        return True
    return False