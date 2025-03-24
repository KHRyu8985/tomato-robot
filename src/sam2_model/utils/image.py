import cv2
import numpy as np

def show_image_masks_and_prompts(image, tomato_detection, borders=True):
    """
    Show the image with the masks and the prompts

    image: image of the image
    tomato_detection: list of dictionaries, each containing 'bbox' and 'mask'
    borders: if True, show the borders of the mask
    Returns:
        np.ndarray: image with masks and prompts visualized using cv2
    """
    image_cv2 = image.copy()
    colors = [
        (180, 180, 255),
        (255, 222, 173),
        (200, 255, 200),
        (230, 190, 230),
        (153, 255, 255),
        (159, 192, 255),
        (255, 235, 215)
    ]
    
    combined_mask = np.zeros((image_cv2.shape[0], image_cv2.shape[1]), dtype=bool)
    
    # Integrate all masks
    for detection in tomato_detection:
        mask = detection['mask']
        
        if mask.ndim == 4:
            mask = mask.squeeze()
        elif mask.ndim == 3:
            mask = mask.squeeze(0)
            
        if mask.ndim != 2:
            mask = mask.reshape(image_cv2.shape[0], image_cv2.shape[1])
        
        mask = mask.astype(float)
        mask_bool = (mask > 0.5) 
        combined_mask = np.logical_or(combined_mask, mask_bool)  
    
    # Apply dark background once for the integrated mask
    alpha = 0.85 
    combined_mask_3c = np.repeat(combined_mask[:, :, None], 3, axis=2) 
    dark_frame = (image_cv2 * (1.0 - alpha)).astype(np.uint8)
    image_cv2 = np.where(combined_mask_3c, image_cv2, dark_frame)
    
    # Add border, bounding box, and label for each object
    for i, detection in enumerate(tomato_detection):
        mask = detection['mask']
        bbox = detection['bbox']
        object_id = detection['id']
        color = colors[i % len(colors)]
        
        if mask.ndim == 4:
            mask = mask.squeeze()
        elif mask.ndim == 3:
            mask = mask.squeeze(0)
            
        if mask.ndim != 2:
            mask = mask.reshape(image_cv2.shape[0], image_cv2.shape[1])
        
        mask = mask.astype(float)
        
        if borders:
            binary_mask = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(image_cv2, contours, -1, (255, 255, 255), thickness=2)
        
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = bbox[2], bbox[3]
        cv2.rectangle(image_cv2, (int(x0), int(y0)), (int(x1), int(y1)), color=color, thickness=2)
        
        x_min, y_min, x_max, y_max = bbox
        text = f'tomato {object_id}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            image_cv2, 
            (int(x_min), int(y_min) - text_size[1] - 10), 
            (int(x_min) + text_size[0], int(y_min) - 5), 
            color=color, 
            thickness=-1 
        )
        cv2.putText(
            image_cv2, 
            text, 
            (int(x_min), int(y_min) - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (50, 50, 50), 
            2, 
            cv2.LINE_AA
        )

    return image_cv2
