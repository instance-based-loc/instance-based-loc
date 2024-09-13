import numpy as np
import imageio
import cv2
from ultralytics import YOLO

model = YOLO('yolov8x.pt')

# Define class names for COCO dataset
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

def detect_objects(image, conf_thresh=0.1, image_size_thresh=0.01):
    # Perform object detection
    results = model(image)
    
    # Retrieve bounding boxes from the results
    boxes = results[0].boxes
    detections = []

    # Get image dimensions
    height, width, _ = image.shape
    image_area = height * width

    # Iterate over each box
    for box in boxes:
        # Extract the bounding box coordinates, confidence, and class ID
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Convert to numpy array
        conf = box.conf.cpu().item()                # Confidence score
        cls_id = int(box.cls.cpu().item())          # Class ID as an integer
        cls_name = COCO_CLASSES[cls_id]              # Get the class name

        # Calculate bounding box area
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Filter by confidence threshold and bounding box area
        if conf > conf_thresh and bbox_area > image_size_thresh * image_area:
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class_id': cls_id,
                'class_name': cls_name
            })
    
    return detections

def visualize_and_save(image, detections, output_path):
    # Convert image to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        confidence = detection['confidence']
        class_id = detection['class_id']
        class_name = detection["class_name"]

        # Draw bounding box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        
        # Display class name and confidence on the bounding box
        label = f'{class_name} ({confidence:.2f})'
        cv2.putText(image_bgr, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the visualized image
    cv2.imwrite(output_path, image_bgr)
