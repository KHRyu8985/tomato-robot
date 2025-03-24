import cv2 as cv
import numpy as np

def visualize_mask_comparison(mask1, mask2, original_image=None):
    """
    visualize the comparison of two masks
    
    Args:
        mask1 (np.ndarray): first mask (hand-pointed part)
        mask2 (np.ndarray): second mask (tomato mask)
        original_image (np.ndarray, optional): original image. if not provided, use black background
    
    Returns:
        np.ndarray: mask comparison visualization image
    """
    # check the dimension of the mask and adjust it
    if mask1.ndim > 2:
        mask1 = mask1.squeeze()
    if mask2.ndim > 2:
        mask2 = mask2.squeeze()

    h, w = mask1.shape[:2]
    if original_image is None:
        background = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        background = cv.resize(original_image, (w, h))
    
    # prepare the result image with 4 panels (2x2 grid)
    # [original image | mask1]
    # [mask2       | intersection]
    result = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    
    # panel 1: original image
    result[:h, :w] = background.copy()
    
    # panel 2: mask1 (red)
    mask1_vis = background.copy()
    mask1_vis[mask1 > 0] = (0, 0, 255)  # red
    result[:h, w:w*2] = mask1_vis
    
    # panel 3: mask2 (green)
    mask2_vis = background.copy()
    mask2_vis[mask2 > 0] = (0, 255, 0)  # green
    result[h:h*2, :w] = mask2_vis
    
    # panel 4: intersection (blue)
    intersection = np.logical_and(mask1 > 0, mask2 > 0)
    intersection_vis = background.copy()
    intersection_vis[intersection] = (255, 0, 0)  # blue
    result[h:h*2, w:w*2] = intersection_vis
    
    # add text
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(result, "Original Image", (10, 30), font, 1, (255, 255, 255), 2)
    cv.putText(result, "Hand Mask (Red)", (w+10, 30), font, 1, (255, 255, 255), 2)
    cv.putText(result, "Tomato Mask (Green)", (10, h+30), font, 1, (255, 255, 255), 2)
    cv.putText(result, "Intersection (Blue)", (w+10, h+30), font, 1, (255, 255, 255), 2)
    
    return result

def calculate_iou(mask1, mask2, debug=False, original_image=None):
    """
    calculate the IoU(Intersection over Union) between two masks
    
    Args:
        mask1 (np.ndarray): first mask
        mask2 (np.ndarray): second mask
        debug (bool): whether to activate debug mode
        original_image (np.ndarray, optional): original image for visualization
    
    Returns:
        float: IoU value (0~1)
    """
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    
    if mask1_area == 0 or mask2_area == 0:
        return 0
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = mask1_area + mask2_area - intersection

    # if debug mode, visualize the mask
    if debug:
        vis_image = visualize_mask_comparison(mask1, mask2, original_image)
        cv.imshow("Mask Comparison", vis_image)
        cv.waitKey(1)  # wait for 1ms (update the screen but not wait)
    
    return intersection / union if union > 0 else 0


def find_matching_tomato(new_mask, tomato_detection, iou_threshold=0.8, debug=False, original_image=None):
    """
    find the most matching tomato with the mask pointed by hand
    
    Args:
        new_mask (np.ndarray): mask pointed by hand
        tomato_detection (list): list of tomato detection results
        iou_threshold (float, optional): minimum IoU value to consider as a match. default is 0.8.
        debug (bool): whether to activate debug mode
        original_image (np.ndarray, optional): original image for visualization
        
    Returns:
        tuple: (matched tomato ID, IoU value) if no matched tomato, return (None, 0)
    """
    if new_mask is None or tomato_detection is None:
        return None, 0
    
    if hasattr(new_mask, 'squeeze'):
        new_mask = new_mask.squeeze()
    
    max_iou = 0
    matched_tomato_id = None
    
    for tomato in tomato_detection:
        tomato_mask = tomato['mask']
        if hasattr(tomato_mask, 'squeeze'):
            tomato_mask = tomato_mask.squeeze()

        # convert to binary mask
        tomato_mask_bin = (tomato_mask > 0.5).astype(np.uint8)
        
        # calculate IoU
        iou = calculate_iou(new_mask, tomato_mask_bin, debug=debug, original_image=original_image)
        
        if iou > max_iou:
            max_iou = iou
            matched_tomato_id = tomato['id']
    
    # return the matched tomato if IoU is greater than the threshold
    if max_iou > iou_threshold:
        return matched_tomato_id, max_iou
    else:
        return None, max_iou