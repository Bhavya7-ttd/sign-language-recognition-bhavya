import cv2
import numpy as np

def draw_skeleton_canvas(hand, frame_shape, offset=29, target_size=(400, 400)):
    """
    Standardized hand landmark preprocessing and skeleton canvas drawing.
    Converts hand landmarks into a 400x400x3 white canvas image with green finger lines
    and red joint markers, matching the model training input format.
    
    Args:
        hand (dict): Hand dictionary from cvzone.HandTrackingModule (contains 'bbox', 'lmList', 'type').
        frame_shape (tuple): Shape of the input camera frame (height, width, channels).
        offset (int): Bounding box margin offset. Default 29.
        target_size (tuple): Output image dimensions (width, height). Default (400, 400).
        
    Returns:
        tuple: (white_canvas, pts_processed)
            - white_canvas (np.ndarray): 400x400x3 uint8 image fed into CNN model.
            - pts_processed (list): Processed 21 landmark [x, y, z] coordinates in crop space.
    """
    x, y, w, h = hand['bbox']
    frame_h, frame_w = frame_shape[:2]
    
    # Safe crop bounds calculation
    y1 = max(0, y - offset)
    y2 = min(frame_h, y + h + offset)
    x1 = max(0, x - offset)
    x2 = min(frame_w, x + w + offset)
    
    if (y2 - y1) <= 0 or (x2 - x1) <= 0:
        white_empty = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        return white_empty, []

    pts_original = hand['lmList']
    pts = []
    for pt in pts_original:
        px = pt[0] - x1
        py = pt[1] - y1
        pz = pt[2] if len(pt) > 2 else 0
        pts.append([int(px), int(py), int(pz)])

    # Centering offsets on target canvas
    os_val = int(((target_size[0] - w) // 2) - 15)
    os1_val = int(((target_size[1] - h) // 2) - 15)

    # Left-hand mirroring handling
    is_left_hand = (hand.get("type", "Right") == "Left")
    if is_left_hand:
        for i in range(21):
            canvas_x = pts[i][0] + os_val
            mirrored_canvas_x = target_size[0] - canvas_x
            pts[i][0] = mirrored_canvas_x - os_val

    # Generate white background canvas
    white = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255

    # Draw hand skeleton lines (green, thickness 3)
    for t in range(0, 4, 1):
        cv2.line(white, (pts[t][0] + os_val, pts[t][1] + os1_val), 
                 (pts[t + 1][0] + os_val, pts[t + 1][1] + os1_val), (0, 255, 0), 3)
    for t in range(5, 8, 1):
        cv2.line(white, (pts[t][0] + os_val, pts[t][1] + os1_val), 
                 (pts[t + 1][0] + os_val, pts[t + 1][1] + os1_val), (0, 255, 0), 3)
    for t in range(9, 12, 1):
        cv2.line(white, (pts[t][0] + os_val, pts[t][1] + os1_val), 
                 (pts[t + 1][0] + os_val, pts[t + 1][1] + os1_val), (0, 255, 0), 3)
    for t in range(13, 16, 1):
        cv2.line(white, (pts[t][0] + os_val, pts[t][1] + os1_val), 
                 (pts[t + 1][0] + os_val, pts[t + 1][1] + os1_val), (0, 255, 0), 3)
    for t in range(17, 20, 1):
        cv2.line(white, (pts[t][0] + os_val, pts[t][1] + os1_val), 
                 (pts[t + 1][0] + os_val, pts[t + 1][1] + os1_val), (0, 255, 0), 3)

    cv2.line(white, (pts[5][0] + os_val, pts[5][1] + os1_val), (pts[9][0] + os_val, pts[9][1] + os1_val), (0, 255, 0), 3)
    cv2.line(white, (pts[9][0] + os_val, pts[9][1] + os1_val), (pts[13][0] + os_val, pts[13][1] + os1_val), (0, 255, 0), 3)
    cv2.line(white, (pts[13][0] + os_val, pts[13][1] + os1_val), (pts[17][0] + os_val, pts[17][1] + os1_val), (0, 255, 0), 3)
    cv2.line(white, (pts[0][0] + os_val, pts[0][1] + os1_val), (pts[5][0] + os_val, pts[5][1] + os1_val), (0, 255, 0), 3)
    cv2.line(white, (pts[0][0] + os_val, pts[0][1] + os1_val), (pts[17][0] + os_val, pts[17][1] + os1_val), (0, 255, 0), 3)

    # Draw joint dots (red circles, radius 2)
    for i in range(21):
        cv2.circle(white, (pts[i][0] + os_val, pts[i][1] + os1_val), 2, (0, 0, 255), 1)

    return white, pts
