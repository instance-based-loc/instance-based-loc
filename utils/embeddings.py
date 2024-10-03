import open_clip
import torch
# import torchvision.transforms as transforms
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
import cv2
from transformers import Dinov2Model, AutoImageProcessor

# global clip_model, clip_preprocess, dino_model, transform_dino
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP Model Loading and Preprocessing
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k"
    )
clip_model = clip_model.to(device)

dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
dino_model = dino_model.to(device)
dino_model.eval()

# Define the image transformation pipeline
transform_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

# Load the pre-trained ViT-Base model and feature extractor
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = vit_model.to(device)
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')


def get_all_clip_embeddings(**kwargs) -> torch.Tensor:
    """
    The function `get_all_clip_embeddings` takes a list of images and returns a list of CLIP embeddings
    for each image.

    :return: A list of CLIP embeddings for each image in the input list of images.
    """
    images = [kwargs["current_obj_grounded_img"]]
    device = kwargs.get("device", "cpu")

    image_clip = Image.fromarray(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    input_clip = clip_preprocess(image_clip).unsqueeze(0).to(device)

    # Calculate CLIP embeddings
    with torch.no_grad():
        clip_features = clip_model.encode_image(input_clip)

    clip_features /= clip_features.norm(dim=-1, keepdim=True)

    return clip_features.squeeze(0)
    

def get_all_dino_embeddings(**kwargs) -> torch.Tensor:
    """
    The function `get_all_dino_embeddings` processes a list of images to obtain DINO embeddings for each
    image.
    
    :return: The function `get_all_dino_embeddings` returns a list of embeddings for each input image in
    the `images` list.
    """
    images = [kwargs["current_obj_grounded_img"]]
    device = kwargs.get("device", "cpu")

    dino_img = Image.fromarray(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB) )
    dino_image_tensor = transform_dino(images=dino_img, return_tensors="pt").pixel_values.to(device)

    # Get the feature embedding from the model
    with torch.no_grad():
        dino_features_cls = dino_model(dino_image_tensor).last_hidden_state[:, 0]

    return dino_features_cls.squeeze(0)


def get_all_vit_embeddings(**kwargs) -> torch.Tensor:
    """
    The function `get_all_vit_embeddings` processes a list of images to obtain ViT embeddings for each
    image.
    
    :return: The function `get_all_vit_embeddings` returns a list of embeddings for each input image in
    the `images` list.
    """
    images = [kwargs["current_obj_grounded_img"]]
    device = kwargs.get("device", "cpu")


    image = Image.fromarray(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))

    # Preprocess the image (resize, normalize, etc.)
    inputs = vit_feature_extractor(images=image, return_tensors="pt").to(device)

    # Pass the preprocessed image through the model
    with torch.no_grad():
        outputs = vit_model(**inputs)

    # Extract the [CLS] token embedding (first token)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding.squeeze(0)


from dator_wrapper import load_model, get_model_input
dator_model = load_model('/home2/aneesh.chavan/instance-based-loc/dator/dator_best_tum.pth')
dator_model.eval()

def get_dator_embeddings(**kwargs) -> torch.Tensor: 
    """
    returns the dator embeddings for the list of images  
    """
    with torch.no_grad():
        images = kwargs["current_obj_grounded_img"]

        bb = kwargs["current_obj_bounding_box"]
        full_depth_image = kwargs["full_depth_image"]

        depth_img = full_depth_image[int(bb[1]):int(bb[3]),
                                    int(bb[0]):int(bb[2])]

        rgb_t, depth_t = get_model_input(images, depth_img)
        emb = dator_model(rgb_t, depth_t).detach().squeeze()

        return emb