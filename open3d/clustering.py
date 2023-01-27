import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

color_map = plt.get_cmap('tab20')


def read_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd


def read_mesh(path):
    return o3d.io.read_triangle_mesh(path)


def read_pcd(path):
    return o3d.io.read_point_cloud(path)


def segment_plane(pcd: o3d.geometry.PointCloud,
                  distance_threshold: float = 0.01
                  ) -> List[o3d.geometry.PointCloud]:

    print(pcd)

    plane_equitation, ind = pcd.segment_plane(distance_threshold=distance_threshold,
                                              ransac_n=3,
                                              num_iterations=1000)

    [a, b, c, d] = plane_equitation
    print(f'Plane equation: {a}x + {b}y + {c}z + {d} = 0')

    on_plane = pcd.select_by_index(ind)
    on_plane.paint_uniform_color([1, 0, 0])

    return [pcd.select_by_index(ind, invert=True), on_plane]


def split_clusters(pcd: o3d.geometry.PointCloud,
                   eps: float = 0.02,
                   min_points: int = 100,
                   ):

    clusters = []

    cl = np.array(pcd.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=True))
    max_clusters = np.max(cl)
    print(f'point cloud has {max_clusters + 1} clusters')
    print(f'cluster labels: {np.unique(cl)}')
    print(f'cluster sizes: {np.bincount(cl + 1)}')

    for i in range(max_clusters + 1):
        cluster = pcd.select_by_index(np.where(cl == i)[0])
        cluster.paint_uniform_color(color_map(i / (max_clusters + 1))[:3])
        clusters.append(cluster)

    # sort clusters by size
    clusters.sort(key=lambda x: len(x.points), reverse=True)

    return clusters

def statistical_outlier_removal(pcd: o3d.geometry.PointCloud,
                                nb_neighbors: int = 20,
                                std_ratio: float = 2.0,
                                ) -> List[o3d.geometry.PointCloud]:

    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])

    return [inlier_cloud, outlier_cloud]

def radius_outlier_removal(pcd: o3d.geometry.PointCloud,
                           nb_points: int = 16,
                           radius=0.05) -> List[o3d.geometry.PointCloud]:

    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=radius)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])

    return [inlier_cloud, outlier_cloud]

def segment_plane(pcd: o3d.geometry.PointCloud,
                  distance_threshold: float = 0.01
                  ) -> List[o3d.geometry.PointCloud]:

    print(pcd)

    plane_equitation, ind = pcd.segment_plane(distance_threshold=distance_threshold,
                                              ransac_n=3,
                                              num_iterations=1000)

    [a, b, c, d] = plane_equitation
    print(f'Plane equation: {a}x + {b}y + {c}z + {d} = 0')

    on_plane = pcd.select_by_index(ind)
    on_plane.paint_uniform_color([1, 0, 0])

    return [pcd.select_by_index(ind, invert=True), on_plane]


def segment_planes(pcd: o3d.geometry.PointCloud,
                   iterations: int = 3,
                   distance_threshold: float = 0.05):
    planes = []
    rest = pcd

    for i in range(iterations):

        rest, plane = segment_plane(rest, distance_threshold)
        colors = color_map(i / iterations)
        # TODO
        plane.paint_uniform_color(list(colors[:3]))
        planes.append(plane)

    return [planes, rest]







def main():
    num_blocks = 4
    pcd_path = pathlib.Path('/home/lucymendoza/ply_files/testscan1.ply')

    if not pcd_path.exists():
        print('pcd file not found')
    pcd = read_ply(str(pcd_path))
    print('pcd read successfully')
    o3d.visualization.draw_geometries([pcd])

    # segment ground plane
    pcd, plane = segment_plane(pcd, 0.003)
    o3d.visualization.draw_geometries([pcd, plane])

    # cluster
    clusters = split_clusters(pcd, eps=0.01, min_points=200)
    clusters = clusters[:num_blocks]
    o3d.visualization.draw_geometries(clusters)

    # outlier removal
    outliers = []
    for cluster in clusters:
        cluster, outlier_s = statistical_outlier_removal(cluster)
        outliers.append(outlier_s)
    outlier_removal = clusters + outliers
    o3d.visualization.draw_geometries(outlier_removal)

    # cluster bounding box
    bb = []
    for cluster in clusters:
         bb.append(o3d.geometry.OrientedBoundingBox.create_from_points(cluster.points))
    cluster_bb = clusters + bb
    o3d.visualization.draw_geometries(cluster_bb)

    # segment planes in clusters
    cl_planes = []
    for cluster in clusters:
        planes, rest = segment_planes(cluster, 4, 0.002)
        cl_planes.append(planes)
    flat_list = [item for sublist in cl_planes for item in sublist]
    o3d.visualization.draw_geometries(flat_list)

    # down sample cluster planes
    for idx, cl in enumerate(cl_planes):
         cl_planes[idx] = [plane.voxel_down_sample(0.005) for plane in cl]
    flat_list = [
         item for sublist in cl_planes for item in sublist]
    o3d.visualization.draw_geometries(flat_list)

    # extract pose
    centers = []
    bbs = []
    for cl in cl_planes:
        centers.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=cl[0].get_center()))
        bbs.append(o3d.geometry.OrientedBoundingBox.create_from_points(cl[0].points))
    o3d.visualization.draw_geometries(centers + bbs + flat_list

    # build mesh
    cluster_planes_meshes = []
    for cl in cl_planes:
        mesh_planes = []
        for plane in cl:
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            #     plane, alpha=0.001)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(plane,
                                                                                   o3d.utility.DoubleVector(
                                                                                       [0.001, 0.005]))
            mesh = mesh.compute_vertex_normals()
            mesh_planes.append(mesh)
        cluster_planes_meshes.append(mesh_planes)
    flat_list = [item for sublist in cluster_planes_meshes for item in sublist]
    o3d.visualization.draw_geometries(flat_list)

    # smooth mesh
    for idx, cl in enumerate(cluster_planes_meshes):
        for jdx, mesh in enumerate(cl):
            mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
            mesh = mesh.compute_vertex_normals()
            cluster_planes_meshes[idx][jdx] = mesh
    flat_list = [item for sublist in cluster_planes_meshes for item in sublist]
    o3d.visualization.draw_geometries(flat_list)

  







if _name_ == "_main_":
    main()