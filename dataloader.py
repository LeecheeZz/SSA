import os
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader

from collections import defaultdict

class Dataloader_University_cluster(Dataset):
    def __init__(self, root, transforms, names=['satellite', 'street', 'drone', 'google']):
        super(Dataloader_University_cluster).__init__()

        self.transforms_drone_street = transforms['train']
        self.transforms_satellite = transforms['satellite']
        self.root = root
        self.names = names
        
        self.cluster_num = 4
        self.sample_num = 2

        print('Sample for train')
        print('cluster_num:', self.cluster_num, ' | cluster_num:', self.sample_num)

        dict_path = {}
        for name in names:
            # dict_ = {}
            dict_ = defaultdict(list)
            i = 0
            # if name == 'satellite':
            #     root = '/home/jllin/data/XiangAn/XiangAn_3line_train01_test2_res240_Satellite_mannulShift/train'
            # else:
            #     root = self.root
            for cls_name in sorted(os.listdir(os.path.join(root, name))):
                cluster_name = str(int(i / self.cluster_num))
                if i % self.cluster_num < self.sample_num:
                    img_list = os.listdir(os.path.join(root, name, cls_name))
                    img_path_list = [os.path.join(root, name, cls_name, img) for img in img_list]
                    dict_[cluster_name].extend(img_path_list)
                i = i + 1

            dict_path[name] = dict_
            # dict_path[name+"/"+cls_name] = img_path_list

        cls_names = list(dict_path[names[0]].keys())
        # cls_names = os.listdir(os.path.join(root, names[0]))
        cls_names.sort()
        map_dict = {i:cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        # self.index_cls_nums = 2

    def sample_from_cls(self, name, cls_num):
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path, 1)[0]
        img = Image.open(img_path)
        return img

    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        img = self.sample_from_cls("satellite", cls_nums)
        img_s = self.transforms_satellite(img)

        img = self.sample_from_cls("drone",cls_nums)
        img_d = self.transforms_drone_street(img)
        return img_s, img_s, img_d, index

    def __len__(self):
        return len(self.cls_names)