import os
from collections import Counter

import cv2
import numpy as np
import pykitti
import skimage
from tqdm import tqdm
import glob

from ekf.kitti_utils import generate_depth_map, sub2ind

base_path = '/mnt/nas_8/datasets/kitti_odometry/dataset/train/'
sequence = '00'
output_path = '/mnt/nas_7/group/guojun/kitti_depth/'

# data = pykitti.odometry(base_path, sequence, frames=range(10))
data = pykitti.odometry(base_path, sequence)

# velo2cam = data.calib.T_cam0_velo
# R_cam2rect = np.eye(4)
# R_cam2rect[:3, :3] = data.calib.P_rect_00[:3, :3]
# P_rect_20 = data.calib.P_rect_20
# P_rect_02 = np.linalg.pinv(P_rect_20)
# P_velo2im = np.dot(np.dot(P_rect_02, R_cam2rect), velo2cam)

P_velo2im = data.calib.T_cam2_velo[:3]
im_shape = data.get_cam2(0).size[::-1]


def apply_colormap(data):
    data = data.copy()
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = np.clip(data, 0, 1)
    data = (data * 255).astype(np.uint8)
    data = cv2.applyColorMap(data, cv2.COLORMAP_TWILIGHT)
    return data

def genDepth(i):
    velo = data.get_velo(i)

    velo = velo[velo[:, 0] >= 0, :]
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im = velo_pts_im @ data.calib.K_cam0.T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0])
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1])
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0
    depth_gt = skimage.transform.resize(
        depth, im_shape, order=0, preserve_range=True, mode='constant')

    cv2.imwrite(output_path + '/frame-{:0>6}.depth.png'.format(i), depth_gt)

def genColor(i):
    color = data.get_cam2(i)

    color_np = np.array(color)
    color_cv2 = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path + '/frame-{:0>6}.color.jpg'.format(i), color_cv2)

def genPose(i):
    scale = 0.3
    pose = data.poses[i]
    pose[:3, 3] = pose[:3, 3] * scale

    with open(output_path + "/frame-{:0>6}.pose.txt".format(i), "w") as output_file:
        for row in pose:
            for num in row:
                output_file.write("{:.6f} ".format(num))
            output_file.write("\n")
    output_file.close()

if __name__ == '__main__':

    for i in tqdm(range(len(data))):

        genDepth(i)
        # genColor(i)
        # genPose(i)




        # # apply color map to a gray image
        # depth_colored = apply_colormap(depth_gt)
        #
        # # blending two images
        # ratio = 0.2
        # vis = color_cv2 * ratio + depth_colored * (1-ratio)
        #
        # # cv2.imwrite(f'/home/shenyichen/Graduate/23-3-20-SimpleRecon/simplerecon/tmp/{i:04d}.png', depth_color)
        #
        # cv2.imwrite(f'/home/shenyichen/Graduate/23-3-20-SimpleRecon/simplerecon/tmp/{i:04d}.png', vis)
        # # cv2.waitKey(0)

    # print(data.calib.K_cam2)
