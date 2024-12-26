import torch
import torch.utils.data
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
from detectron2.data import transforms as T
from torch.utils.data import Dataset
from copy import deepcopy


def read_list(path):
    data_list = []
    with open(path, 'r', encoding='utf-8') as infile:
        for name in infile:
            data_name = name.strip('\n').split()[0]
            data_list.append(data_name)
    return data_list


class OccDataset(Dataset):
    def __init__(self, img_folder, occ_folder, ids_path, image_set):
        super().__init__()

        self.resolution = 256
        self.img_folder = img_folder
        self.occ_folder = occ_folder

        # Data Augmentation
        self.augmentation = make_transforms(image_set)

        # data_split
        self.data_list = read_list(f'{ids_path}/{image_set}_list.txt')
        valid_data_list = [x.split('.')[0] for x in os.listdir(self.img_folder)]
        self.data_list = list(filter(lambda x: x in self.data_list, valid_data_list))
        
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        name = self.data_list[index]
        occ_path = f'{self.occ_folder}/{name}.npy'
        occ_data = np.load(occ_path, allow_pickle=True)
        
        img_path = f'{self.img_folder}/{name}.png'
        img = np.array(Image.open(img_path))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w, h = img.shape

        data = {}
        data["name"] = name
        data["height"] = h
        data['width'] = w
        
        if self.augmentation is None:
            data['image'] = (1 / 255) * torch.as_tensor(np.array(np.expand_dims(img, 0)))
            queries = []
            occs = []
            for room_occ in occ_data:
                query = room_occ['query']
                occ = room_occ['occ']
                queries.append(query)
                occs.append(occ)
            queries = np.stack(queries, axis=0)
            occs = np.stack(occs, axis=0)
            queries = (queries + 0.5) / self.resolution - 0.5
            queries = torch.as_tensor(queries, dtype=torch.float32)
            data['occ'] = torch.as_tensor(occs, dtype=torch.float32)
            data['query'] = torch.cat([queries, torch.ones(queries.shape[0], queries.shape[1], 1)], dim=2)
        else:
            aug_input = T.AugInput(img)
            transforms = self.augmentation(aug_input)
            image = aug_input.image
            data['image'] = (1 / 255) * torch.as_tensor(np.array(np.expand_dims(image, 0)))
            queries = []
            occs = []
            for room_occ in occ_data:
                query = room_occ['query']
                occ = room_occ['occ']
                query = transforms.apply_coords(query)
                query[query <= 0] = 0
                query[query >= 255.0] = 255.0
                queries.append(query)
                occs.append(occ)
            queries = np.stack(queries, axis=0)
            occs = np.stack(occs, axis=0)
            queries = (queries + 0.5) / self.resolution - 0.5
            queries = torch.as_tensor(queries, dtype=torch.float32)
            data['occ'] = torch.as_tensor(occs, dtype=torch.float32)
            data['query'] = torch.cat([queries, torch.ones(queries.shape[0], queries.shape[1], 1)], dim=2)

        return data
   

def make_transforms(image_set):
    if image_set == 'train':
        return T.AugmentationList([
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomRotation([0.0, 90.0, 180.0, 270.0], expand=False, center=None, sample_style="choice")
        ])
    else:
        return None
    

def build(image_set, args):
    dataset = OccDataset(args.img_folder, args.occ_folder, args.ids_path, image_set)
    return dataset
