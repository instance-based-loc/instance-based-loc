from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import numpy as np
from functools import lru_cache

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    The data loader should be able to handle both environment and evaluation datasets.
    """

    def __init__(self, data_path: str, evaluation_indices: Optional[Tuple[int]]):
        """
        Initialize the data loader.

        Args:
            data_path (str): The location of the data
            evaluation_indices (Optional[Tuple[int]]): The indices that are evaluated for localization
                If the data diretcory by itself only contains only the data that should be used for evaluation
                    then handle it in the inheriting class.
        """
        self.data_path = data_path
        self.evaluation_indices = evaluation_indices

    @property
    @lru_cache(maxsize=None)
    def environment_indices(self) -> Tuple[int, ...]:
        """       
        This property should return the indices that are used for the environment dataset.
        
        Returns:
            Tuple[int, ...]: The indices for the environment dataset.
        """
        return self._get_environment_indices()

    @abstractmethod
    def _get_environment_indices(self) -> Tuple[int, ...]:
        """
        Abstract method that must be implemented by inheriting classes.
        
        This method should return the indices that are used for the environment dataset.
        
        Returns:
            Tuple[int, ...]: The indices for the environment dataset.
        """
        pass

    @abstractmethod
    def get_image_data(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Get the RGB image, depth map, and pose at the specified index.
        
        Args:
            index (int): Index of the image.
        
        Returns:
            Tuple containing:
            - RGB image (np.ndarray)
            - Optional depth map (np.ndarray)
            - Pose (np.ndarray)
        """
        pass

    @abstractmethod
    def get_pointcloud(self, bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        """
        Get the pointcloud data, optionally truncated by a bounding box.
        
        Args:
            bounding_box (Optional[Dict[str, Tuple[float, float]]]): Bounding box to truncate the pointcloud, with keys 'x', 'y', 'z' and values as tuples (min, max).
        
        Returns:
            np.ndarray: Truncated pointcloud data.
        """
        pass

    @abstractmethod
    def get_pointcloud_from_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        Get the pointcloud's portion that is viewed from the current pose. 
        NOTE - not a depth map of the current viewed position, but truncate the view frustum to a reasonable limit. 
        
        Args:
            pose: the current pose (of generally the camera)
        
        Returns:
            np.ndarray: Truncated pointcloud data viewed from the current pose. 
        """
        pass
