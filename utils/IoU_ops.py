import numpy
import open3d as o3d
import sys, os
sys.path.append(os.path.join(os.getcwd(), "Objectron"))

def calculate_3d_IoU(pcd1, pcd2):
    """
    Calculates the 3D Intersection over Union (IoU) between two 3D point clouds.

    Parameters:
    - pcd1 (o3d.geometry.PointCloud): open3d point clouds
    - pcd2 (o3d.geometry.PointCloud): open3d point clouds

    Returns:
    - IoU (float): 3D Intersection over Union between the two point clouds.
    """

    pcd1 = np.asarray(pcd1.points).T
    pcd2 = np.asarray(pcd2.points).T

    try:
        bb1_min = pcd1.min(axis=-1)
        bb1_max = pcd1.max(axis=-1)

        bb2_min = pcd2.min(axis=-1)
        bb2_max = pcd2.max(axis=-1)

        overlap_min_corner = np.stack([bb1_min, bb2_min], axis=0).max(axis=0)
        overlap_max_corner = np.stack([bb1_max, bb2_max], axis=0).min(axis=0)

    except:
        return 0

    if (overlap_min_corner > overlap_max_corner).any():
        return 0
    else:
        v = overlap_max_corner - overlap_min_corner
        overlap_volume = v[0] * v[1] * v[2]
        
        bb1 = bb1_max - bb1_min
        bb2 = bb2_max - bb2_min

        v1 = bb1[0] * bb1[1] * bb1[2]
        v2 = bb2[0] * bb2[1] * bb2[2]

        IoU = overlap_volume / (v1 + v2 - overlap_volume)

        return IoU

def calculate_strict_overlap(pcd1, pcd2):
    """
    Calculates the strict overlap between two 3D point clouds.

    Parameters:
    - pcd1 (o3d.geometry.PointCloud): open3d point clouds
    - pcd2 (o3d.geometry.PointCloud): open3d point clouds

    Returns:
    - overlap (float): Strict overlap between the two point clouds.
    """

    pcd1 = np.asarray(pcd1.points).T
    pcd2 = np.asarray(pcd2.points).T


    try:
        bb1_min = pcd1.min(axis=-1)
        bb1_max = pcd1.max(axis=-1)

        bb2_min = pcd2.min(axis=-1)
        bb2_max = pcd2.max(axis=-1)

        overlap_min_corner = np.stack([bb1_min, bb2_min], axis=0).max(axis=0)
        overlap_max_corner = np.stack([bb1_max, bb2_max], axis=0).min(axis=0)
    except:
        return 0

    if (overlap_min_corner > overlap_max_corner).any():
        return 0
    else:
        v = overlap_max_corner - overlap_min_corner
        overlap_volume = v[0] * v[1] * v[2]
        
        bb1 = bb1_max - bb1_min
        bb2 = bb2_max - bb2_min

        v1 = bb1[0] * bb1[1] * bb1[2]
        v2 = bb2[0] * bb2[1] * bb2[2]

        overlap = overlap_volume / min(v1, v2)

        return overlap

def calculate_obj_aligned_3d_IoU(pcd1, pcd2):
    """
    Calculates the 3D Intersection over Union (IoU) between two 3D point clouds. Using Objectrons algorithm
    Uses object aligned bounding boxes isntead of axis aligned

    Parameters:
    - pcd1 (numpy.ndarray): First 3D point cloud represented as a 3xN array.
    - pcd2 (numpy.ndarray): Second 3D point cloud represented as a 3xN array.

    Returns:
    - IoU (float): 3D Intersection over Union between the two point clouds.
    """
    def conv_to_objectron_ordering(v):
        v = sorted(v, key=lambda v: v[2])
        v = sorted(v, key=lambda v: v[1])
        v = sorted(v, key=lambda v: v[0])
        return v

    try:
        bb1 = o3d.geometry.OrientedBoundingBox.create_from_points(
            points=o3d.utility.Vector3dVector(pcd1.T) #, robust=True
        )
        bb2 = o3d.geometry.OrientedBoundingBox.create_from_points(
            points=o3d.utility.Vector3dVector(pcd2.T) #, robust=True
        )
    except:
        return 0

    bb1_vertices = np.zeros((9,3), dtype=np.float32)
    bb1_vertices[0, :] = bb1.get_center()
    bb1c = np.array(bb1.get_box_points())
    bb1_vertices[1:,:] = conv_to_objectron_ordering(bb1c)

    bb2_vertices = np.zeros((9,3), dtype=np.float32)
    bb2_vertices[0, :] = bb2.get_center()
    bb2c = np.array(bb2.get_box_points())
    bb2_vertices[1:,:] = conv_to_objectron_ordering(bb2c)

    w1 = box.Box(vertices=bb1_vertices)
    w2 = box.Box(vertices=bb2_vertices)

    loss = iou.IoU(w1, w2)
    try:
        iou3d = loss.iou()
    except:
        iou3d = 0.

    return iou3d