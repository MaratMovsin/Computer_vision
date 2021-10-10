#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))

def make_mask(image, points, size):
    mask = 255*np.ones(np.array(image).shape).astype(np.uint8)
    for point in points:
        #print(point)
        cv2.circle(mask, (int(point[0]), int(point[1])), size, 0, -1)
    return mask
    
def add_corners(old_corners: FrameCorners, image, feature_params, next_id, mask = None) -> [int, FrameCorners]:
    new_corners_array = cv2.goodFeaturesToTrack(image, mask=mask, **feature_params)
    if new_corners_array is None or feature_params['maxCorners'] == 0:
        return [next_id, old_corners]
    new_corners_array = new_corners_array.reshape((-1, 2))
    #max_id = max(old_corners.ids)
    new_ids = next_id+np.arange(len(new_corners_array))
    new_ids = new_ids.reshape((-1,1))
    ids = np.concatenate((old_corners.ids,new_ids))
    next_id = np.max(ids)+1
    corners_array = np.concatenate((old_corners.points, new_corners_array))
    sizes = np.concatenate((old_corners.sizes, 7*np.ones(len(corners_array)).reshape((-1, 1))))
    corners = FrameCorners(ids, corners_array, sizes)
    return [next_id, corners]

def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    image_0 = (image_0 * 255.0).astype(np.uint8)
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.03,
                       minDistance = 20,
                       blockSize = 7 )
    corners_array = cv2.goodFeaturesToTrack(image_0, mask = None, **feature_params).reshape((-1, 2))
    corners = FrameCorners(np.array(range(len(corners_array))), corners_array, 7*np.ones(len(corners_array)))
    next_id = len(corners_array)
    builder.set_corners_at_frame(0, corners)
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))
    counter = 0
    
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        counter += 1
        image_1 = (image_1 * 255.0).astype(np.uint8)
        corners_array, st, err = cv2.calcOpticalFlowPyrLK(image_0, image_1, corners.points, None, **lk_params,minEigThreshold=0.003)
        if st.sum()!=0:
            filtered_corners_array = corners_array[np.hstack((st,st))==1].reshape(-1,2)
            corners = FrameCorners(corners.ids[st==1],filtered_corners_array,corners.sizes[st==1])
        else:
            counter=0
        
        if counter%5 == 0:
            feature_params['maxCorners'] = int(100-st.sum())
            next_id, corners = add_corners(corners, image_0, feature_params, next_id, mask = make_mask(image_1, corners.points, 50))
        
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
