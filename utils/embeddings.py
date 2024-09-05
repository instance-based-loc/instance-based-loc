import open_clip
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

# global clip_model, clip_preprocess, dino_model, transform_dino
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP Model Loading and Preprocessing
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
clip_model = clip_model.to(device)

# Load the pre-trained DINO model
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

dino_model = dino_model.to(device)
dino_model.eval()

# Define the image transformation pipeline
transform_dino = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Combined Function for Calculating Embeddings
def get_clip_embedding(cropped_img, device="cpu"):
    # Load and preprocess the image for CLIP
    image_clip = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    input_clip = clip_preprocess(image_clip).unsqueeze(0).to(device)

    # Calculate CLIP embeddings
    with torch.no_grad():
        clip_features = clip_model.encode_image(input_clip)

    clip_features /= clip_features.norm(dim=-1, keepdim=True)

    return clip_features

def get_dino_embedding(cropped_img, device="cpu"):
    # Load and preprocess the image for DinoV2
    dino_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    dino_image_tensor = transform_dino(dino_img).unsqueeze(0).to(device)  # Add batch dimension and send to device

    # Get the feature embedding from the model
    with torch.no_grad():
        dino_features = dino_model(dino_image_tensor)[0]

    return dino_features

def get_all_clip_embeddings(**kwargs) -> torch.Tensor:
    """
    The function `get_all_clip_embeddings` takes a list of images and returns a list of CLIP embeddings
    for each image.

    :return: A list of CLIP embeddings for each image in the input list of images.
    """
    images = [kwargs["current_obj_grounded_img"]]
    device = kwargs.get("device", "cpu")

    clip_embeddings = []
    for image in images:
        clip_embedding = get_clip_embedding(image, device)
        clip_embeddings.append(clip_embedding)
    return torch.stack(clip_embeddings)

def get_all_dino_embeddings(**kwargs) -> torch.Tensor:
    """
    The function `get_all_dino_embeddings` processes a list of images to obtain DINO embeddings for each
    image.
    
    :return: The function `get_all_dino_embeddings` returns a list of embeddings for each input image in
    the `images` list.
    """
    images = [kwargs["current_obj_grounded_img"]]
    device = kwargs.get("device", "cpu")

    dino_embeddings = []
    for image in images:
        dino_embedding = get_dino_embedding(image, device)
        dino_embeddings.append(dino_embedding)
    return torch.stack(dino_embeddings)


def get_dator_embeddings(**kwargs) -> torch.Tensor: 
    """
    returns the dator embeddings for the list of images  
    """
    # from dator.dator_wrapper import load_dator, get_single_embedding  
    images = [kwargs["current_obj_grounded_img"]] 
    print(f"{images = }") 
    import sys 
    sys.exit(0) 