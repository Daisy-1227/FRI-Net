import numpy as np
import os
import cv2
from multiprocessing import Process, Queue
import time
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import copy
import os
import matplotlib.pyplot as plt
from PIL import Image


# Modify the path to your own data folder
data_path = "/mnt/d/projects/FRI-Net/FRI-Net/data"

def read_list(image_set, dataset="stru3d"):
    data_list = []
    file_path = f'{data_path}/{dataset}/{image_set}_list.txt'
    with open(file_path, 'r', encoding='utf-8') as infile:
        for name in infile:
            data_name = name.strip('\n').split()[0]
            data_list.append(data_name)
    return data_list


def generate_occ(dataset="stru3d"):
    save_folder = f"{data_path}/{dataset}/occ"
    os.makedirs(save_folder, exist_ok=True)
    if dataset == "stru3d":
        label_files = [f"{data_path}/stru3d/annotations/train.json", 
                    f"{data_path}/stru3d/annotations/val.json", 
                    f"{data_path}/stru3d/annotations/test.json"]

    for ann_file in label_files:
        coco = COCO(ann_file)
        ids = list(sorted(coco.imgs.keys()))
        for id in ids:
            ann_ids = coco.getAnnIds(imgIds=id)
            target = coco.loadAnns(ann_ids)
            target = [t for t in target if t['category_id'] not in [16, 17]]
            file_name = coco.loadImgs(id)[0]['file_name'].split('.')[0]
            occ_data = []
            for room_id, each_room in enumerate(target):
                room_seg = each_room['segmentation'][0]
                room_corners = np.array(room_seg).reshape(-1, 2)

                
                spatial_query = np.mgrid[:256, :256]
                spatial_query = np.moveaxis(spatial_query, 0, -1)
                spatial_query = spatial_query.reshape(-1, 2).astype(np.float32)

                mask = np.zeros((256, 256))
                cv2.fillPoly(mask, [room_corners.astype(np.int32)], 1.)

                spatial_indices = np.round(spatial_query).astype(np.int64)
                spatial_occ = mask[spatial_indices[:, 1], spatial_indices[:, 0]]

                spatial_query = spatial_query.reshape(256, 256, 2)
                spatial_occ = spatial_occ.reshape(256, 256, 1)
                sub_row_indices, sub_col_indices = np.meshgrid(np.arange(1, 256, 4), np.arange(1, 256, 4),
                                                               indexing='ij')
                query = spatial_query[sub_row_indices, sub_col_indices]
                query = query.reshape(64 * 64, 2)
                occ = spatial_occ[sub_row_indices, sub_col_indices]
                occ = occ.reshape(64 * 64, 1)

                room_occ = dict()
                room_occ['query'] = query
                room_occ['occ'] = occ
                occ_data.append(room_occ)
            save_path = f'{save_folder}/{file_name}.npy'
            np.save(save_path, occ_data)
            print(f'generate occ: {save_path}')


def generate_input_img(dataset="stru3d"):
    if dataset == "stru3d":
        density_folder = f"{data_path}/stru3d/density"
        height_folder = f"{data_path}/stru3d/height"
    
    file_list = os.listdir(density_folder)
    file_list = sorted(file_list)
    
    save_folder = f"{data_path}/{dataset}/input"

    os.makedirs(save_folder, exist_ok=True)

    for file in file_list:
        print(f'file_name: {file}')
        density_path = f'{density_folder}/{file}'
        height_path = f'{height_folder}/{file}'
        if not os.path.exists(density_path) or not os.path.exists(height_path):
            continue
        density = cv2.imread(density_path)
        height = cv2.imread(height_path)
        # image_name = file.split('.')[0]
        # cv2.imshow('density', density)
        # cv2.imshow('height', height)
        # cv2.waitKey(0)
        print(f'generate: {file}')
        input_img = np.maximum(density, height)
        cv2.imwrite(f'{save_folder}/{file}', input_img)


if __name__ == '__main__':
    generate_input_img(dataset="stru3d")
    generate_occ(dataset="stru3d")