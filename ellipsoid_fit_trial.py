import numpy as np
import open3d as o3d
from clip_loc.ellipsoid_utils import fit_ellipsoid_to_point_cloud

def create_cylinder(radius, height, resolution, color):
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    mesh_cylinder.compute_vertex_normals()

    pcd_cylinder = mesh_cylinder.sample_points_uniformly(number_of_points=5000)

    colors = np.full((np.asarray(pcd_cylinder.points).shape[0], 3), color, dtype=np.float32)
    pcd_cylinder.colors = o3d.utility.Vector3dVector(colors)

    return pcd_cylinder


# Step 2: Generate random point cloud data
pcl = create_cylinder(radius=1.0, height=2.0, resolution=30, color=[0, 0, 1])

# Step 4: Set the color of the point cloud to green
pcl.paint_uniform_color([0, 1, 0])  # RGB for green

ellipsoid_pcl = fit_ellipsoid_to_point_cloud(pcl)

# Step 9: Set the color of the ellipsoid point cloud to red
ellipsoid_pcl.paint_uniform_color([1, 0, 0])  # RGB for red

# Step 11: Save the visualizations separately
o3d.io.write_point_cloud("./out/points_with_fitted_ellipsoid_pcl.ply", pcl + ellipsoid_pcl)