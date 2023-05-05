# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2
import glob

from dataset.base_dataset import BaseDataset
import json


class kitti(BaseDataset):
    def __init__(self, img_path, depth_path):
        super().__init__()
        self.img_path = img_path
        self.depth_path = depth_path
        self.depth_tag = self.build_list()

    def build_list(self):
        depth_tag = glob.glob(os.path.join(self.depth_path,  "frame-*.depth.png"))
        return depth_tag

    def __len__(self):
        return len(self.depth_tag)

    def __getitem__(self, idx):
        frame_num = self.depth_tag[idx].split('/')[-1].split('-')
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        # filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        class_id = -1

        assert class_id >= 0

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

        # print(image.shape, depth.shape, self.scale_size)

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))

        # print(image.shape, depth.shape, self.scale_size)

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth = depth / 1000.0  # convert in meters

        return {'image': image, 'depth': depth, 'filename': filename, 'class_id': class_id}
