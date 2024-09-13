import numpy as np
import open3d as o3d

def fit_ellipsoid_to_point_cloud(pcl, scaling_factor=1.05, max_iterations=20):
    """
    Fit an ellipsoid to a given point cloud and return the ellipsoid as a point cloud.

    Parameters:
    - pcl: open3d.geometry.PointCloud
    - scaling_factor: float
        Factor by which to scale the ellipsoid to ensure it encloses all points.
    - max_iterations: int
        Number of iterations to refine the ellipsoid fit.

    Returns:
    - ellipsoid_pcl: open3d.geometry.PointCloud
        A point cloud representing the fitted ellipsoid.
    """
    points = np.asarray(pcl.points)

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    covariance_matrix = np.cov(centered_points, rowvar=False)

    U, S, Vt = np.linalg.svd(covariance_matrix)

    axes_lengths = np.sqrt(S)

    axes_lengths *= scaling_factor

    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    
    ellipsoid_points = []
    
    for _ in range(max_iterations):
        ellipsoid_points = []
        for i in range(len(u)):
            for j in range(len(v)):
                x = axes_lengths[0] * np.cos(u[i]) * np.sin(v[j])
                y = axes_lengths[1] * np.sin(u[i]) * np.sin(v[j])
                z = axes_lengths[2] * np.cos(v[j])
                point = np.dot([x, y, z], U.T) + centroid
                ellipsoid_points.append(point)

        ellipsoid_pcl = o3d.geometry.PointCloud()
        ellipsoid_pcl.points = o3d.utility.Vector3dVector(ellipsoid_points)

        if all(np.linalg.norm(np.dot(point - centroid, U) / axes_lengths) <= 1 for point in points):
            break

        axes_lengths *= scaling_factor

    return ellipsoid_pcl
