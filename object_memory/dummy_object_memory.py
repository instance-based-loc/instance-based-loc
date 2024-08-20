from typing import Type
import torch
import numpy as np
import imageio

from .base_object_memory import BaseObjectMemory

class DummyObjectMemory(BaseObjectMemory):
    def _load_rgb_image(self, path):
        return np.asarray(imageio.imread(path))

    def _load_depth_image(self, path):
        return np.load(path)

    def _get_embeddings(self, *args, **kwargs):
        return torch.Tensor([10, 10])

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)