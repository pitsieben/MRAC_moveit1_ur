import pathlib
from typing import List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import open3d as o3d

color_map = plt.get_cmap('tab20')
# conda install pillow matplotlib


def read_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd


pcd_path = pathlib.Path('/home/pit/Downloads/eight.ply')

if not pcd_path.exists():
    print('pcd file not found')
pcd = read_ply(str(pcd_path))

print('pcd read successfully')
# o3d.visualization.draw_geometries([pcd])


# statistical outlier removal
def remove_outliers(pcd, nb_neighbors, std_ratio):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    return cl
# o3d.visualization.draw_geometries([remove_outliers(pcd, 20, 2.0)])


# segment plane
def plane_segmentation(pcd, distance_threshold):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 0.5, 0.5])

    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])

    return inlier_cloud, outlier_cloud


inlier_cloud, outlier_cloud = plane_segmentation(pcd, 0.007)
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

pcb = outlier_cloud
# o3d.visualization.draw_geometries([pcb])

# crop bounding box


def crop_bounding_box(pcd, min_bound, max_bound):
    pcd_crop = pcd.crop(
        o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
    return pcd_crop


pcb = crop_bounding_box(pcb, [-50, -100, 0.005], [90, 100, 100])
# o3d.visualization.draw_geometries([pcb])


# VOXEL DOWN SAMPLING
# def voxel_down_sample(pcd, voxel_size):
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#     return pcd_down
# pcd=voxel_down_sample(pcb, 0.001)
# o3d.visualization.draw_geometries([pcd])

# pcd = voxel_down_sample(pcd, 0.001)


# CONVEX HULL
def convex_hull(pcd):
    chull, chull_indices = pcd.compute_convex_hull()
    chull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(chull)
    chull_ls.paint_uniform_color([1, 0.706, 0])
    return chull_ls


convexhull = convex_hull(pcb)
# o3d.visualization.draw_geometries([convexhull, pcb])


# point cloud to mesh
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcb, o3d.utility.DoubleVector([0.001, 0.005]))
mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])


# filter smooth simple
print("Try to remove the noise")
mesh_simple = mesh.filter_smooth_simple(15)
mesh_simple.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh_simple])

# paint mesh


def paint_mesh(pcd_mesh, color):
    pcd_mesh.paint_uniform_color(color)
    return pcd_mesh


mesh_paint = paint_mesh(mesh_simple, [0.5, 0.5, 0.5])
# o3d.visualization.draw_geometries([mesh_paint])

# mesh edge extraction


def extract_mesh_edges(mesh):
    lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lineset.paint_uniform_color([0, 0, 0])
    return lineset


mesh_edge = extract_mesh_edges(mesh)
o3d.visualization.draw_geometries([mesh_edge, mesh])

# create bounding box


def create_bounding_box(pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.color = (0, 0, 1)
    return bbox


bounding_box = create_bounding_box(pcb)
o3d.visualization.draw_geometries([bounding_box, mesh])

# bounding box dimensions


def bounding_box_dimensions(bbox):
    print(bbox.get_extent())


bounding_box_dimensions(bounding_box)


# visualiza bounding box dimensions

# def estimate_normals(pcd, radius):
#     pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
#     return pcd

# o3d.visualization.draw_geometries([estimate_normals(pcd, 0.01)])
