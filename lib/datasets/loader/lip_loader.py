##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Jianyuan Guo, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import os
import cv2

import numpy as np
from torch.utils import data

from lib.utils.helpers.image_helper import ImageHelper
from lib.extensions.parallel.data_container import DataContainer
from lib.utils.tools.logger import Logger as Log


class LipLoader(data.Dataset):
    def __init__(self, root_dir, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_list, self.label_list, self.edge_list, self.name_list = self.__list_dirs(root_dir, dataset)
        self.root_dir = root_dir
        self.dataset = dataset

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        img_size = ImageHelper.get_size(img)
        labelmap = ImageHelper.read_image(self.label_list[index],
                                          tool=self.configer.get('data', 'image_tool'), mode='P')
        edgemap = ImageHelper.read_image(self.edge_list[index],
                                          tool=self.configer.get('data', 'image_tool'), mode='P')
            
        edgemap[edgemap==255] = 1
        edgemap = cv2.resize(edgemap, (labelmap.shape[-1], labelmap.shape[-2]), interpolation = cv2.INTER_NEAREST)

        if self.configer.exists('data', 'label_list'):
            labelmap = self._encode_label(labelmap)

        if self.configer.exists('data', 'reduce_zero_label') and self.configer.get('data', 'reduce_zero_label') == 'True':
            labelmap = self._reduce_zero_label(labelmap)

        ori_target = ImageHelper.tonp(labelmap)
        ori_target[ori_target == 255] = -1

        if self.aug_transform is not None:
            img, labelmap, edgemap = self.aug_transform(img, labelmap=labelmap, maskmap=edgemap)

        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)
            edgemap = self.label_transform(edgemap)

        meta = dict(
            ori_img_size=img_size,
            border_size=border_size,
            ori_target=ori_target
        )
        return dict(
            img=DataContainer(img, stack=True),
            labelmap=DataContainer(labelmap, stack=True),
            maskmap=DataContainer(edgemap, stack=True),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(self.name_list[index], stack=False, cpu_only=True),
        )

    def _reduce_zero_label(self, labelmap):
        if not self.configer.get('data', 'reduce_zero_label'):
            return labelmap

        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1
        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def _encode_label(self, labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.float32) * 255
        for i in range(len(self.configer.get('data', 'label_list'))):
            class_id = self.configer.get('data', 'label_list')[i]
            encoded_labelmap[labelmap == class_id] = i

        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        label_list = list()
        edge_list = list()
        name_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')
        label_dir = os.path.join(root_dir, dataset, 'label')
        edge_dir = os.path.join(root_dir, dataset, 'edge')
        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        for file_name in os.listdir(label_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
            label_path = os.path.join(label_dir, file_name)
            edge_path = os.path.join(edge_dir, file_name)
            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label Path: {} not exists.'.format(label_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)
            edge_list.append(edge_path)
            name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            image_dir = os.path.join(root_dir, 'val/image')
            label_dir = os.path.join(root_dir, 'val/label')
            edge_dir = os.path.join(root_dir, 'val/edge')
            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                edge_path = os.path.join(edge_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                edge_list.append(edge_path)
                name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'include_atr'):
            image_dir = os.path.join(root_dir, 'atr/image')
            label_dir = os.path.join(root_dir, 'atr/label')
            edge_dir = os.path.join(root_dir, 'atr/edge')
            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                edge_path = os.path.join(edge_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                edge_list.append(edge_path)
                name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'include_cihp'):
            image_dir = os.path.join(root_dir, 'cihp/single_person/image')
            label_dir = os.path.join(root_dir, 'cihp/single_person/label')
            edge_dir = os.path.join(root_dir, 'cihp/single_person/edge')
            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                edge_path = os.path.join(edge_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                edge_list.append(edge_path)
                name_list.append(image_name)

        return img_list, label_list, edge_list, name_list


if __name__ == "__main__":
    pass