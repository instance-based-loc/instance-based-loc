import numpy as np
import matplotlib.pyplot as plt
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    ToPILImage
)
from tqdm import tqdm
from transformers import ViTConfig, ViTModel, ViTForImageClassification
from transformers import AutoImageProcessor, CLIPVisionModel
from peft import LoraConfig, get_peft_model
import numpy as np

class LoraRevolver:
    """
    Loads a base ViT and a set of LoRa configs, allows loading and swapping between them.
    """
    def __init__(self, device, model_checkpoint="google/vit-base-patch16-224-in21k"):
        """
        Initializes the LoraRevolver object.
        
        Parameters:
        - device (str): Device to be used for compute.
        - model_checkpoint (str): Checkpoint for the base ViT model.
        """
        self.device = device
        
        # self.base_model will be augmented with a saved set of lora_weights
        # self.lora_model is the augmented model (NOTE)
        self.base_model = ViTModel.from_pretrained(
            model_checkpoint,
            ignore_mismatched_sizes=True,
        ).to(self.device)
        self.lora_model = self.base_model

        # image preprocessors the ViT needs
        self.image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)
        self.normalize = Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
        self.train_transforms = Compose(
            [
                RandomResizedCrop(self.image_processor.size["height"]),
                RandomHorizontalFlip(),
                ToTensor(),
                self.normalize,
            ]
        )
        self.test_transforms = Compose(
            [
                Resize(self.image_processor.size["height"]),
                CenterCrop(self.image_processor.size["height"]),
                ToTensor(),
                self.normalize,
            ]
        )

        # stored lora_configs, ready to be swapped in
        # only expects store lora_checkpoints.pt objects created by this class
        self.ckpt_library = {}


    def load_lora_ckpt_from_file(self, config_path, name):
        """
        Load a LoRa config from a saved file.
        
        Parameters:
        - config_path (str): Path to the LoRa config file.
        - name (str): Name to associate with the loaded config.
        """
        ckpt = torch.load(config_path)
        try:
            self.ckpt_library[str(name)] = ckpt
            del self.lora_model
            self.lora_model = get_peft_model(self.base_model,
                                                ckpt["lora_config"]).to(self.device)
            self.lora_model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        except:
            print("Lora checkpoint invalid")
            raise IndexError

    def encode_image(self, **kwargs):
        """
        Use the current LoRa model to encode a batch of images.
        
        Parameters:
        - imgs (list): List of images to encode.
        
        Returns:
        - emb (torch.Tensor): Encoded embeddings for the input images.
        """
        imgs = [kwargs["current_obj_grounded_img"]]

        with torch.no_grad():
            if isinstance(imgs[0], np.ndarray):
                img_batch = torch.stack([Compose([ToPILImage(),
                                                  self.test_transforms])(i) for i in imgs])
            else:
                img_batch = torch.stack([self.test_transforms(i) for i in imgs])
            # if len(img.shape) == 3:
            #     img = img.unsqueeze(0)    # if the image is unbatched, batch it
            emb = self.lora_model(img_batch.to(self.device), output_hidden_states=True).last_hidden_state[:,0,:]
        
        # fix to port code over
        if emb.shape[0] == 1:
            emb = emb[0]

        return emb
    
    def train_current_lora_model(self):
        """
        Train the current LoRa model.
        """
        pass

    def save_lora_ckpt(self):
        """
        Save the current LoRa model checkpoint.
        """
        pass
