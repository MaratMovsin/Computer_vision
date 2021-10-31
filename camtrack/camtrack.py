#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    build_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    Correspondences,
    TriangulationParameters
)


def build_correspondences_3d_2d(cloud: PointCloudBuilder, corners: FrameCorners,
                          ids_to_remove=None) -> Correspondences:
    ids_1 = cloud.ids.flatten()
    ids_2 = corners.ids.flatten()
    _, (indices_1, indices_2) = snp.intersect(ids_1, ids_2, indices=True)
    corrs = Correspondences(
        ids_1[indices_1],
        cloud.points[indices_1],
        corners.points[indices_2]
    )
    if ids_to_remove is not None:
        corrs = _remove_correspondences_with_ids(corrs, ids_to_remove)
    return corrs



def find_camera_position(correspondences: Correspondences,
                         intrinsic_mat: np.ndarray)\
        -> np.ndarray:
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
            correspondences.points_1,
            correspondences.points_2,
            intrinsic_mat,
            None)
    return  [rodrigues_and_translation_to_view_mat3x4(rvec, tvec), inliers.shape[0]]


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    corners_0 = corner_storage[0]
    point_cloud_builder = PointCloudBuilder()
    
    correspondence = build_correspondences(corner_storage[known_view_1[0]],
                                         corner_storage[known_view_2[0]])
    triangulation_parametrs = TriangulationParameters(0.65, 0, 0)
    points3d, ids, median_cos = triangulate_correspondences(
        correspondence,
        pose_to_view_mat3x4(known_view_1[1]),
        pose_to_view_mat3x4(known_view_2[1]),
        intrinsic_mat,
        triangulation_parametrs
    )
    point_cloud_builder.add_points(ids, points3d)
    print('Calculating cloud using frames {0} and {1}'.format(known_view_1[0], known_view_2[0]))
    print('Median cosinus is {0}'.format(median_cos))
    print('Current size of cloud is {0}'.format(point_cloud_builder.ids.shape[0]))
    
    step_size = 10
    
    correspondence = build_correspondences_3d_2d(point_cloud_builder,
                                                 corner_storage[0])
    view_mats[0], inliers_size = find_camera_position(correspondence, intrinsic_mat)
    print('Calculating camera position on frame {0}'.format(0))
    print('Number of inliers is {0}'.format(inliers_size))
    
    for i in range(known_view_2[0]):
        if i%step_size != 0:
            correspondence = build_correspondences_3d_2d(point_cloud_builder,
                                                 corner_storage[i])
            view_mats[i], inliers_size = find_camera_position(correspondence, intrinsic_mat)
            print('Calculating camera position on frame {0}'.format(i))
            print('Number of inliers is {0}'.format(inliers_size))
        else:
            j = i+step_size
            if j >= frame_count:
                j = frame_count-1
            correspondence = build_correspondences_3d_2d(point_cloud_builder,
                                                 corner_storage[j])
            view_mats[j], inliers_size = find_camera_position(correspondence, intrinsic_mat)
            print('Calculating camera position on frame {0}'.format(j))
            print('Number of inliers is {0}'.format(inliers_size))
            correspondence = build_correspondences(
                corner_storage[i],
                corner_storage[j]
            )
            points3d, ids, median_cos = triangulate_correspondences(
                correspondence,
                view_mats[i],
                view_mats[j],
                intrinsic_mat,
                triangulation_parametrs
            )
            point_cloud_builder.add_points(ids, points3d)
            print('Calculating cloud using frames {0} and {1}'.format(i, j))
            print('Median cosinus is {0}'.format(median_cos))
            print('Current size of cloud is {0}'.format(point_cloud_builder.ids.shape[0]))
    
    step_size = 5
    for i in range(known_view_2[0],frame_count-1):
        if (i-10)%step_size != 0:
            correspondence = build_correspondences_3d_2d(point_cloud_builder, corner_storage[i])
            view_mats[i], inliers_size = find_camera_position(correspondence, intrinsic_mat)
            print('Calculating camera position on frame {0}'.format(i))
            print('Number of inliers is {0}'.format(inliers_size))
        else:
            j = i+step_size
            if j >= frame_count:
                j = frame_count-1
            correspondence = build_correspondences_3d_2d(point_cloud_builder,
                                                 corner_storage[j])
            view_mats[j], inliers_size = find_camera_position(correspondence, intrinsic_mat)
            print('Calculating camera position on frame {0}'.format(j))
            print('Number of inliers is {0}'.format(inliers_size))
            correspondence = build_correspondences(
                corner_storage[i-step_size],
                corner_storage[j]
            )
            points3d, ids, median_cos = triangulate_correspondences(
                correspondence,
                view_mats[i-step_size],
                view_mats[j],
                intrinsic_mat,
                triangulation_parametrs
            )
            point_cloud_builder.add_points(ids, points3d)
            print('Calculating cloud using frames {0} and {1}'.format(i, j))
            print('Median cosinus is {0}'.format(median_cos))
            print('Current size of cloud is {0}'.format(point_cloud_builder.ids.shape[0]))
    
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud

if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
