from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import numpy as np
from functools import lru_cache
import open3d as o3d

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
    def get_image_data(self, index: int) -> Tuple[str, Optional[str], np.ndarray]:
        """
        Get the RGB image path, depth map path, and pose at the specified index.
        
        Args:
            index (int): Index of the image.
        
        Returns:
            Tuple containing:
            - RGB image path (str)
            - Optional depth map (str)
            - Pose (np.ndarray)
        """
        pass

    @abstractmethod
    def get_pointcloud(self, bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> o3d.geometry.PointCloud:
        """
        Get the pointcloud data, optionally truncated by a bounding box.
        
        Args:
            bounding_box (Optional[Dict[str, Tuple[float, float]]]): Bounding box to truncate the pointcloud, with keys 'x', 'y', 'z' and values as tuples (min, max).
        
        Returns:
            np.ndarray: Truncated pointcloud data.
        """
        pass

    @abstractmethod
    def get_visible_pointcloud(self, pose: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Filters the point cloud to include only points visible from a given pose.

        Args:
            pose (np.ndarray): The pose containing translation and quaternion (x, y, z, qx, qy, qz, qw).

        Returns:
            np.ndarray: The filtered point cloud.
        """
        pass