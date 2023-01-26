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


pcd_path = pathlib.Path('/home/pit/Downloads/8test.ply')

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
inlier_cloud, outlier_cloud = plane_segmentation(pcd, 0.003)
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

pcb = outlier_cloud
# o3d.visualization.draw_geometries([pcb])

def voxel_down_sample(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down


# o3d.visualization.draw_geometries([voxel_down_sample(pcd, 0.005)])
pcd = voxel_down_sample(pcd, 0.001)

# cluster
def split_clusters(pcd, eps, min_points):
    cl, ind = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    max_clusters = np.max(cl)
    print(f'point cloud has {max_clusters + 1} clusters')
    print(f'cluster labels: {np.unique(cl)}')
    print(f'cluster sizes: {np.bincount(cl + 1)}')

    clusters = []
    for i in range(max_clusters + 1):
        cluster = pcd.select_by_index(np.where(cl == i)[0])
        cluster.paint_uniform_color(color_map(i / (max_clusters + 1))[:3])
        clusters.append(cluster)

    # sort clusters by size
    clusters.sort(key=lambda x: len(x.points), reverse=True)
    return clusters
clusters = split_clusters(pcd, 0.01, 50)
clusters = clusters[1:]
o3d.visualization.draw_geometries(clusters)




# # DBSCAN CLUSTERING
# def dbscan_clustering(pcd, eps, min_points):
#     cl, ind = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
#     max_clusters = np.max(cl)
#     print(f'point cloud has {max_clusters + 1} clusters')
#     print(f'cluster labels: {np.unique(cl)}')
#     print(f'cluster sizes: {np.bincount(cl + 1)}')

#     clusters = []
#     for i in range(max_clusters + 1):
#         cluster = pcd.select_by_index(np.where(cl == i)[0])
#         cluster.paint_uniform_color(color_map(i / (max_clusters + 1))[:3])
#         clusters.append(cluster)

#     # sort clusters by size
#     clusters.sort(key=lambda x: len(x.points), reverse=True)

#     return clusters



# VOXEL DOWN SAMPLING




# vertex normal estimation


# def estimate_normals(pcd, radius):
#     pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
#     return pcd

# o3d.visualization.draw_geometries([estimate_normals(pcd, 0.01)])

# crop point cloud


# def crop_point_cloud(pcd, min_bound, max_bound):
#     pcd_crop = pcd.crop(
#         o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
#     return pcd_crop

# o3d.visualization.draw_geometries([crop_point_cloud(pcd, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])])

# paint point cloud


# def paint_point_cloud(pcd, color):
#     pcd.paint_uniform_color(color)
#     return pcd

# o3d.visualization.draw_geometries([paint_point_cloud(pcd, [1, 0.706, 0])])

# CONVEX HULL


def convex_hull(pcd):
    chull, chull_indices = pcd.compute_convex_hull()
    chull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(chull)
    chull_ls.paint_uniform_color([1, 0.706, 0])
    return chull_ls


convex_hull(pcd)
# o3d.visualization.draw_geometries([convex_hull(pcd)])

# PLANE SEGMENTATION





# inlier_cloud, outlier_cloud = plane_segmentation(pcd, 0.0001)
# o3d.visualization.draw_geometries(
#     [inlier_cloud, outlier_cloud, convex_hull(pcd)])

# point cloud edge extraction


# def edge_extraction(pcd, radius):
#     pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
#     pcd.orient_normals_towards_camera_location()
#     pcd.orient_normals_consistent_tangent_plane(30 * np.pi / 180)

#     pcd_edge = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#         pcd, o3d.utility.DoubleVector([radius, radius * 2]))
#     return pcd_edge


# o3d.visualization.draw_geometries(
#     [edge_extraction(pcd, 0.01), inlier_cloud, outlier_cloud])

# Triangle mesh
print("Testing mesh in Open3D...")
pcd_mesh = o3d.io.read_triangle_mesh(str(pcd_path))
print(pcd_mesh)
print("")
print("Computing normal and rendering it.")
pcd_mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([pcd_mesh])

# filter smooth simple
print("Try to remove the noise")
pcd_mesh = pcd_mesh.filter_smooth_simple(10)
# o3d.visualization.draw_geometries([pcd_mesh])

# paint mesh


def paint_mesh(pcd_mesh, color):
    pcd_mesh.paint_uniform_color(color)
    return pcd_mesh


pcd_mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([paint_mesh(pcd_mesh, [0.5, 0.5, 0.5])])

# mesh edge extraction




