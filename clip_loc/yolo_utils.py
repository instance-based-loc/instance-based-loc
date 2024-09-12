import numpy as np
import imageio
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load the YOLOv8 model


def detect_objects(image):
    results = model(image)  # Perform inference on the image
    
    # Extract bounding boxes and labels
    boxes = results[0].boxes
    detections = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.xyxy[0].tolist() + [box.conf[0].item(), box.cls[0].item()]

        if conf > 0.1:
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class_id': cls
            })
    
    return detections

def visualize_and_save(image, detections, output_path):
    # Convert image to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        confidence = detection['confidence']
        class_id = int(detection['class_id'])

        # Draw bounding box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(image_bgr, f'ID: {class_id}, Conf: {confidence:.2f}', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the visualized image
    cv2.imwrite(output_path, image_bgr)
