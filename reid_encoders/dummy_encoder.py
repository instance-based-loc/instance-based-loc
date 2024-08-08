from .base_encoder import BaseEncoder
import numpy as np

class DummyEncoder(BaseEncoder):
    def __init__(self): 
        super().__init__()
    
    def get_embedddings(self, img):
        return np.zeros(10)

