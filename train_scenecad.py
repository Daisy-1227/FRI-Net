import os
import sys
import torch.nn as nn
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import tqdm
import math
from pathlib import Path
import sys
import cv2
from datasets.occ_data import build as build_dataset
from models.fri_net import build as build_model


def get_args_parser():
    parser = argparse.ArgumentParser('FRI-Net', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--lr_drop', default=[420], type=list)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')

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
    
    # pre-trained room-wise encoder checkpoint
    parser.add_argument('--encoder_path', default='./checkpoints/pretrained_room_wise_encoder.pth', type=str)

    # room-wise decoder parameter
    parser.add_argument('--phase', default=0, type=int)
    parser.add_argument('--num_horizontal_line', default=256, type=int)
    parser.add_argument('--num_vertical_line', default=256, type=int)
    parser.add_argument('--num_diagnoal_line', default=256, type=int)
    parser.add_argument('--num_convex', default=64, type=int)

    # dataset parameters
    parser.add_argument('--dataset_name', default='scenecad')
    parser.add_argument('--img_folder', default='./data/scenecad/input', type=str)
    parser.add_argument('--occ_folder', default='./data/scenecad/occ', type=str)
    parser.add_argument('--ids_path', default='./data/scenecad/', type=str)


    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--output_dir', default='checkpoints', type=str)
    parser.add_argument('--job_name', default='train_scenecad', type=str)
    parser.add_argument('--resume', default=None, type=str)
    return parser


def main(args):

    print(args)
    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load Model
    model, criterion = build_model(args=args)
    # model = nn.DataParallel(model)
    model = model.cuda()
    # model.cuda()
    criterion.cuda()

    if args.phase == 0:
        '''
         Since the room-wise encoder is built upon roomformer, we find that fine-tuning the transformer models from roomformer outperforms training from scratch.
         We therefore initialize the weights of room-wise encoder with a pretrained roomformer checkpoint, which we remove the "dice loss" and the "Iterative polygon refinement"
        '''
        # # Load pretrained base model for room-wise encoder
        encoder_weight = torch.load(args.encoder_path, map_location='cpu')['model']
        state_dict = model.room_wise_encoder.state_dict()
        for name, param in encoder_weight.items():
            if name in state_dict:
                state_dict[name].copy_(param)
        model.room_wise_encoder.load_state_dict(state_dict)
        print(f'Load pretrained ckpt from {args.encoder_path}')
    
    if args.phase == 1:
        ckpt_path = f"{args.output_dir}/checkpoint_0.pth"
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=True)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        print(f'load ckpts from {ckpt_path}')
        # Update the weights of Binary_matrix
        model.room_wise_decoder.update_weights()
    elif args.phase == 2:
        ckpt_path = f"{args.output_dir}/checkpoint_1.pth"
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=True)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        print(f'load ckpts from {ckpt_path}')

    # Load data
    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('val', args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                 pin_memory=True)
    
    # Build optimizer
    
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out


    param_dicts = [
        {
            "params":
                [p for n, p in model.room_wise_encoder.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names) 
                    and not match_name_keywords(n, args.lr_linear_proj_names)
                    and p.requires_grad],
            "lr": args.lr * 0.1,
        },
        {
            "params": [p for n, p in model.room_wise_encoder.named_parameters() if
                        match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone * 0.1,
        },
        {
            "params": [p for n, p in model.room_wise_encoder.named_parameters() if
                        match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult * 0.1,
        },
        {
            "params": [p for p in model.room_wise_decoder.parameters() if p.requires_grad],
            "lr": args.lr,
        },
    ]


    # 创建优化器
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    if args.resume is not None:
        print(f"resume")
        ckpt_path = f"{args.output_dir}/checkpoint_{args.phase}.pth"
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=True)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        print(f'load ckpts from {ckpt_path}')

        args.start_epoch = checkpoint['epoch'] + 1
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = False
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        train_stats= train_one_epoch(model, criterion, data_loader_train, optimizer, epoch, args)
        lr_scheduler.step()
        eval_stats = eval(model, criterion, data_loader_val, epoch, args)
        if args.output_dir:
            output_dir = Path(args.output_dir)
            checkpoint_paths = [output_dir / f'checkpoint_{args.phase}.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) in args.lr_drop or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_{args.phase}.pth')
            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'eval_{k}': v for k, v in eval_stats.items()},
                        'epoch': epoch}
            print(f'write info into {output_dir}/log.txt')
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


def train_one_epoch(model, criterion, data_loader, optimizer, epoch, args):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for batched_inputs in metric_logger.log_every(data_loader, print_freq, header):
        images = [x["image"].cuda() for x in batched_inputs]
        queries = [x['query'].cuda() for x in batched_inputs]
        occ = [x['occ'].cuda() for x in batched_inputs]
        pad_queries = pad_gt_queries(queries, args.num_rooms)
        images = torch.stack(images)
        outputs = model(images, pad_queries)

        target = {
            "occ": occ
        }
        
        loss_dict = criterion(outputs, target)

        loss, loss_occ = loss_dict['loss'], loss_dict['loss_occ']

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss.item())
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss)
        metric_logger.update(loss_occ=loss_occ['shape_occ'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


def eval(model, criterion, data_loader, epoch, args):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for batched_inputs in metric_logger.log_every(data_loader, print_freq, header):
        images = [x["image"].cuda() for x in batched_inputs]
        queries = [x['query'].cuda() for x in batched_inputs]
        occ = [x['occ'].cuda() for x in batched_inputs]
        pad_queries = pad_gt_queries(queries, args.num_rooms)        
        outputs = model(images, pad_queries)
        target = {
            "occ": occ
        }
    
        loss_dict = criterion(outputs, target)

        loss, loss_occ = loss_dict['loss'], loss_dict['loss_occ']

        metric_logger.update(loss=loss)
        metric_logger.update(loss_occ=loss_occ['shape_occ'])

    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


def pad_gt_queries(queries, num_rooms):
    pad_queries = []
    for query in queries:
        num_query = query.size(1)
        pad_query = torch.zeros((num_rooms, num_query, 3), dtype=torch.float32).cuda()
        for _ in range(num_rooms):
            pad_query[_] = query[0]
        pad_queries.append(pad_query)
    pad_queries = torch.stack(pad_queries)
    return pad_queries


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FRI-Net training script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = f'{args.output_dir}/{args.job_name}'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

