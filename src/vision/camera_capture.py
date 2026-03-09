import cv2
import os

def capture_image(save_dir="data/captures"):
    """
    Captures a single image from the configured webcam and saves it.
    Returns the file path of the captured image.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "capture.jpg")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
        
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read frame from webcam.")
        
    cv2.imwrite(file_path, frame)
    cap.release()
    
    return file_path

def open_camera_stream():
    """
    Opens the camera stream natively for preview purposes.
    Mainly used for debugging or direct stream hooks.
    """
    cap = cv2.VideoCapture(0)
    return cap
