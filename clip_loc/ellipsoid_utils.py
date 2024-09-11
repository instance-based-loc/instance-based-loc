import numpy as np
import open3d as o3d
from microstructpy.geometry import Ellipsoid
import os

def fit_ellipsoid_to_point_cloud(pcl):
    """
    Fit an ellipsoid to a given point cloud and return the ellipsoid as a point cloud.

    Parameters:
    - pcl: open3d.geometry.PointCloud

    Returns:
    - ellipsoid_pcl: open3d.geometry.PointCloud
        A point cloud representing the fitted ellipsoid.
    """
    points = np.asarray(pcl.points)

    # Step 2: Fit an ellipsoid to the points
    ellipsoid = Ellipsoid().best_fit(points)

    # Step 3: Access the ellipsoid parameters
    center = ellipsoid.center
    axes = ellipsoid.axes

    # Step 4: Create a mesh for the ellipsoid for visualization
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + axes[0] * np.outer(np.cos(u), np.sin(v))
    y = center[1] + axes[1] * np.outer(np.sin(u), np.sin(v))
    z = center[2] + axes[2] * np.outer(np.ones(np.size(u)), np.cos(v))

    # Create a mesh from the ellipsoid data
    ellipsoid_mesh = o3d.geometry.TriangleMesh()
    ellipsoid_mesh.vertices = o3d.utility.Vector3dVector(np.vstack((x.flatten(), y.flatten(), z.flatten())).T)

    # Create triangles for the ellipsoid mesh
    triangles = []
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            idx1 = i * len(v) + j
            idx2 = (i + 1) * len(v) + j
            idx3 = (i + 1) * len(v) + (j + 1)
            idx4 = i * len(v) + (j + 1)
            triangles.append([idx1, idx2, idx3])
            triangles.append([idx1, idx3, idx4])
            
    ellipsoid_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Step 5: Create a point cloud from the ellipsoid mesh vertices
    ellipsoid_pcl = o3d.geometry.PointCloud()
    ellipsoid_pcl.points = ellipsoid_mesh.vertices

    return ellipsoid_pcl
