import torch
import clip
import numpy as np
from numpy.linalg import norm

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-L/14", device=device)

import cv2  # Make sure to import OpenCV for resizing

def encode_object_images(image: np.ndarray, detections: list):
    """
    Encodes each detected object's image using CLIP.

    Parameters:
    - image: The original image as a NumPy array.
    - detections: A list of detections containing bounding boxes.

    Returns:
    - A list of CLIP embeddings for each detected object.
    """
    object_embeddings = []
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        # Crop the detected object from the original image
        object_image = image[y1:y2, x1:x2]

        # Resize the cropped image to 224x224
        object_image = cv2.resize(object_image, (224, 224))

        # Convert the cropped image to a suitable format for CLIP
        object_image = np.array(object_image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        object_image = np.transpose(object_image, (2, 0, 1))  # Change to (C, H, W)

        # Encode the image using CLIP
        object_embedding = model.encode_image(torch.from_numpy(object_image).unsqueeze(0).to(device))
        object_embeddings.append(object_embedding.cpu().detach().numpy()[0])
    
    return object_embeddings

def embed_text_list(text_lst: list[str]):
    text = clip.tokenize(text_lst).to(device)
    text_embedding = model.encode_text(text)

    text_embedding_cpu = text_embedding.to("cpu")

    return text_embedding_cpu.detach().numpy()

def embed_text(text: str):
    return embed_text_list([text])[0]

def clip_similarity(vector_1: np.ndarray, vector_2: np.ndarray):
    return  np.dot(vector_1, vector_2) / (norm(vector_1) * norm(vector_2))