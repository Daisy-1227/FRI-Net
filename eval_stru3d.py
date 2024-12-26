import os
import sys
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch.utils
import torch.utils.data
import wandb
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from util.poly_ops import pad_gt_polys
import tqdm
import math
from pathlib import Path
import sys
import cv2
from shapely.geometry import Polygon, MultiPolygon
from util.bspt_2d import digest_bsp
import matplotlib.pyplot as plt
import time
from multiprocessing import Process, Queue, Pool
import queue
from train_stru3d import pad_gt_queries
from s3d_floorplan_eval.planar_graph_utils import get_regions_from_pg
from datasets.occ_data import build as build_dataset
from models.fri_net import build as build_model
from shapely.ops import unary_union
from util.postprocess_utils import postprocess


def get_args_parser():
    parser = argparse.ArgumentParser('FRI-Net', add_help=False)
    
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)

    # backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # room-wise encoder params
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=7, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=800, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_rooms', default=20, type=int,
                        help="Number of maximum number of rooms")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--query_pos_type', default='sine', type=str, choices=('static', 'sine', 'none'),
                        help="Type of query pos in decoder - \
                           1. static: same setting with DETR and Deformable-DETR, the query_pos is the same for all layers \
                           2. sine: since embedding from reference points (so if references points update, query_pos also \
                           3. none: remove query_pos")
    
    # room-wise decoder parameter
    parser.add_argument('--phase', default=2, type=int)
    parser.add_argument('--num_horizontal_line', default=256, type=int)
    parser.add_argument('--num_vertical_line', default=256, type=int)
    parser.add_argument('--num_diagnoal_line', default=256, type=int)
    parser.add_argument('--num_convex', default=64, type=int)

    # dataset parameters
    parser.add_argument('--dataset_name', default='stru3d')
    parser.add_argument('--img_folder', default='./data/stru3d/input', type=str)
    parser.add_argument('--occ_folder', default='./data/stru3d/occ', type=str)
    parser.add_argument('--ids_path', default='./data/stru3d/', type=str)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--checkpoint', default="./checkpoints/pretrained_ckpt.pth", type=str)
    return parser

def main(args):
    print(args)

    # Load Model
    model, criterion = build_model(args=args)
    model.cuda()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=True)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    print(f'load ckpts from {args.checkpoint}')

    # Load test data
    dataset_eval = build_dataset('test', args)
    sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch

    data_loader_eval = DataLoader(dataset_eval, args.batch_size, sampler=sampler_eval,
                                  drop_last=False, collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                  pin_memory=True)

    save_folder = f'./results'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    
    evaluate(model, data_loader_eval, save_folder, args)


@torch.no_grad()
def evaluate(model, data_loader, save_folder, args, save_primitive=True):
    model.eval()
    npy_folder = f"{save_folder}/npy"
    vis_folder = f"{save_folder}/vis"
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder, exist_ok=True)
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder, exist_ok=True)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Build input query grid, if need acceleration, set resolution to 64
    resolution = 256
    mgrid = np.zeros([resolution, resolution, 2], dtype=np.float32)
    if save_primitive:
        for x in range(resolution):
            for y in range(resolution):
                mgrid[x, y, :] = [y, x]
    else:
        for x in range(resolution):
            for y in range(resolution):
                mgrid[x, y, :] = [x, y]
    mgrid = (mgrid + 0.5) / resolution - 0.5
    coords = np.reshape(mgrid, [1, resolution * resolution, 2])
    # expand xy to xy1
    coords = np.concatenate([coords, np.ones([1, resolution * resolution, 1], np.float32)], axis=2)

    for batched_inputs in metric_logger.log_every(data_loader, 10, header):

        queries = torch.as_tensor(np.tile(coords, (len(batched_inputs), args.num_rooms, 1, 1)), dtype=torch.float32).cuda()
        images = [x["image"].cuda() for x in batched_inputs]
        img_names = [x["name"] for x in batched_inputs]
        pad_queries = pad_gt_queries(queries, args.num_rooms)
        outputs = model(images, pad_queries)
        pred_lines, convex_occ, shape_occ = outputs['line_param'], outputs['convex_occ'], outputs['pred_occ']
        
        bs = pred_lines.shape[0]

        for b_i in range(bs):
            img = cv2.imread(f"{args.img_folder}/{img_names[b_i]}.png")
            scene_occ = shape_occ[b_i]
            convex_occ_per_scene = convex_occ[b_i]
            pred_lines_per_scene = pred_lines[b_i]
            room_num = 0
            img_name = img_names[b_i]

            # select the valid room index
            valid_indices = torch.where(abs(torch.min(scene_occ, axis=1)[0]) < 0.01)[0].detach().cpu().numpy()
            save_polygons = []
            for room_i in valid_indices:
                room_occ = scene_occ[room_i]

                # print correspond room_occ field
                output_img = np.clip(
                    np.resize(room_occ.squeeze(0).detach().cpu().numpy(), [resolution, resolution]) * 256,
                    0,
                    255).astype(np.uint8)
                output_img = 255 - output_img
                # save_img_path = os.path.join(save_folder, f'{img_name}_{room_num}_implicit_field.png')
                # cv2.imwrite(save_img_path, output_img)
                convex_occ_per_room = convex_occ_per_scene[room_i]
                pred_lines_per_room = pred_lines_per_scene[room_i]
                # room_name = f'{img_name}_{room_i}.png'

                pred_lines_per_room = pred_lines_per_room.detach().cpu().numpy()
                convex_occ_per_room = convex_occ_per_room.detach().cpu().numpy()
                binary_mat = model.room_wise_decoder.binary_matrix.detach().cpu().numpy()

                num_axis_lines = args.num_horizontal_line + args.num_vertical_line
                num_non_axis_lines = args.num_diagnoal_line

                axis_binary_mat = binary_mat[:num_axis_lines, :]
                non_axis_binary_mat = binary_mat[num_axis_lines:, :]

                pred_axis_line = pred_lines_per_room[:, :num_axis_lines]
                pred_non_axis_line = pred_lines_per_room[:, num_axis_lines:]

                convex_occ_axis_line = convex_occ_per_room[:, :args.num_convex]
                convex_occ_non_axis_line = convex_occ_per_room[:, args.num_convex:]

                image_out_size = 256
                polygon_list = []

                binary_mat_lst = [axis_binary_mat, non_axis_binary_mat]
                pred_lines_lst = [pred_axis_line, pred_non_axis_line]
                convex_occ_lst = [convex_occ_axis_line, convex_occ_non_axis_line]

                for _ in range(2):
                    binary_mat = binary_mat_lst[_]
                    pred_lines_per_room = pred_lines_lst[_]
                    convex_occ_per_room = convex_occ_lst[_]

                    line_num, convex_num = binary_mat.shape[0], binary_mat.shape[1]
                    convex_list = []
                    color_idx_list = []

                    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                                (0, 255, 255)]
                    convex_occ_per_room = convex_occ_per_room < 0.01
                    convex_out_sum = np.sum(convex_occ_per_room, axis=1)

                    for i in range(convex_num):
                        slice_i = convex_occ_per_room[:, i]
                        if np.max(slice_i) > 0:
                            if np.min(
                                    convex_out_sum - slice_i * 2) >= 0:  # if this convex is redundant, i.e. the convex is inside the shape
                                convex_out_sum = convex_out_sum - slice_i
                            else:
                                box = []
                                for j in range(line_num):
                                    if binary_mat[j, i] > 0.01:
                                        a = -pred_lines_per_room[0, j]
                                        b = -pred_lines_per_room[1, j]
                                        d = -pred_lines_per_room[2, j]
                                        box.append([a, b, d])
                                if len(box) > 0:
                                    convex_list.append(np.array(box, np.float32))
                                    color_idx_list.append(i % len(color_list))
  
                    # convert convexes to room polygon
                    for i in range(len(convex_list)):
                        vg, tg = digest_bsp(convex_list[i], bias=0)
                        vertices = ((np.array(vg) + 0.5) * 256).astype(np.float32)
                        edges = np.array(tg)
                        # remove duplicate vertices and edges
                        unique_data = get_unique_data(vertices, edges)
                        # get room polygons
                        primitive = get_regions_from_pg(unique_data, corner_sorted=True)
                        if len(primitive) == 0:
                            continue
                        else:
                            primitive = primitive[0]
                        polygon_list.append(Polygon(primitive))
                        cg = color_list[room_num % len(color_list)]
                        for j in range(len(tg)):
                            x1 = ((vg[tg[j][0]][1] + 0.5) * image_out_size).astype(np.int32)
                            y1 = ((vg[tg[j][0]][0] + 0.5) * image_out_size).astype(np.int32)
                            x2 = ((vg[tg[j][1]][1] + 0.5) * image_out_size).astype(np.int32)
                            y2 = ((vg[tg[j][1]][0] + 0.5) * image_out_size).astype(np.int32)
                    cg = color_list[room_num % len(color_list)]
                if len(polygon_list) == 1:
                    polygon = polygon_list[0]
                    save_polygons.append(polygon)
                elif len(polygon_list) > 1:
                    polygon = Polygon()
                    for _ in range(len(polygon_list)):
                        if polygon_list[_].is_valid:
                            polygon = polygon.union(polygon_list[_])
                    save_polygons.append(polygon)
            room_polys = postprocess(save_polygons)
            for room in room_polys:
                for _ in range(room.shape[0]):
                    cv2.circle(img, room[_].astype(np.uint8), 3, [255, 255, 255], -1)
                    cv2.circle(img, room[(_+1)%room.shape[0]].astype(np.uint8), 3, [255, 255, 255], -1)
                    cv2.line(img, room[_].astype(np.uint8), room[(_+1)%room.shape[0]].astype(np.uint8), [255, 255, 255], 2)
            save_img_path = f'{vis_folder}/{img_name}.png'
            save_path = f'{npy_folder}/{img_name}.npy'
            print(f'generate: {img_name}')
            cv2.imwrite(save_img_path, img)
            np.save(save_path, room_polys)


def visualize(polygon_list, img):
    import copy
    vis = copy.deepcopy(img)
    for polygon in polygon_list:
        room_polys = np.array(polygon.exterior.coords, dtype=np.int32)
        cv2.polylines(vis, [room_polys],
             isClosed=True, color=[0, 255, 0], thickness=1)
        for corner in room_polys:
            cv2.circle(vis, corner, 2, [0, 0, 255], -1)
    cv2.imshow('floorplan', vis)
    cv2.waitKey(0)

    
def get_unique_data(corners, edges):
    unique_points, unique_indices, inverse_indices = np.unique(corners, axis=0, return_index=True, return_inverse=True)
    unique_src_indices = [list() for idx in range(len(unique_indices))]
    for src_idx, unique_idx in enumerate(inverse_indices):
        unique_src_indices[unique_idx].append(src_idx)
    unique_edges = []
    visit_mat = np.zeros((len(unique_indices), len(unique_indices)))
    for edge in edges:
        corner0, corner1 = edge[0], edge[1]
        unique_idx0, unique_idx1 = -1, -1
        for unique_idx, src_indices in enumerate(unique_src_indices):
            if corner0 in src_indices:
                unique_idx0 = unique_idx
                break
        for unique_idx, src_indices in enumerate(unique_src_indices):
            if corner1 in src_indices:
                unique_idx1 = unique_idx
                break
        if visit_mat[unique_idx0][unique_idx1] == 0 or visit_mat[unique_idx1][unique_idx0] == 0:
            unique_edges.append([unique_idx0, unique_idx1])
            visit_mat[unique_idx0][unique_idx1] = 1
            visit_mat[unique_idx1][unique_idx0] = 1
    unique_data = {
        'corners': unique_points,
        'edges': np.array(unique_edges, dtype=np.int32)
    }
    return unique_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser('FRI-Net eval script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)