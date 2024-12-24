import torch
import torch.nn.functional as F
import copy
import math

from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from util.misc import NestedTensor, nested_tensor_from_tensor_list, interpolate
from models.ops.modules import MSDeformAttn
from models.transformer import build_transformer
from models.backbone import build_backbone


class room_wise_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Define backbone
        self.backbone = build_backbone(args)

        # Define Transformer
        self.transformer = build_transformer(args)

        # Define learnable_query
        self.num_queries = args.num_queries
        self.num_rooms = args.num_rooms
        self.num_queries_per_room = self.num_queries // self.num_rooms
        hidden_dim = self.transformer.d_model

        self.query_embed = nn.Embedding(self.num_queries, 2)
        self.tgt_embed = nn.Embedding(self.num_queries, hidden_dim)

        # Define other params
        self.num_feature_levels = args.num_feature_levels
        num_backbone_outs = len(self.backbone.strides)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = self.backbone.num_channels[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        for _ in range(self.num_feature_levels - num_backbone_outs):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
            ))
            in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
    

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        bs = samples.tensors.shape[0]
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight
        tgt_embeds = self.tgt_embed.weight

        room_codes = self.transformer(srcs, masks, pos, query_embeds, tgt_embeds)
        room_codes = room_codes.reshape(bs, self.num_rooms, self.num_queries_per_room, -1)
        return room_codes