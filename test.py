from utils.fpfh_register import get_SVD_transform
import numpy as np

# source_points = np.array([
#     [1.0, 2.0, 3.0],
#     [4.0, 5.0, 6.0],
#     [7.0, 8.0, 9.0]
# ])

# target_points = np.array([
#     [1.0, 2.0, 3.0],
#     [4.0, 5.0, 6.0],
#     [7.0, 8.0, 9.0],
# ])

# print(get_SVD_transform(source_points, target_points))

source_points = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

target_points = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])

print(get_SVD_transform(source_points, target_points))

source_points = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

target_points = np.array([
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
    [-1.0, -1.0, 1.0]
])

print(get_SVD_transform(source_points, target_points))

source_points = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

target_points = np.array([
    [1.5, 2.5, 3.5],
    [4.5, 5.5, 6.5],
    [7.5, 8.5, 9.5]
])

print(get_SVD_transform(source_points, target_points))


exit(0)