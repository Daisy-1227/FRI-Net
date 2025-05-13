import torch
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.encoder import room_wise_encoder
from models.decoder import room_wise_decoder
from models.occ_matcher import build_matcher


class FRINet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 定义Encoder and Decoder
        self.phase = args.phase
        self.room_wise_encoder = room_wise_encoder(args=args)
        if self.phase == 0:
            num_line = args.num_horizontal_line + args.num_vertical_line
        else:
            num_line = args.num_horizontal_line + args.num_vertical_line + args.num_diagnoal_line
        
        self.room_wise_decoder = room_wise_decoder(num_line=num_line, args=args)

    """
    Input: 
        samples: input images
        queries: query points
    Output:
        line_param
        occupancy
    """
    def forward(self, samples, queries):
        room_codes = self.room_wise_encoder(samples)
        outputs = self.room_wise_decoder(room_codes, queries)
        return outputs
    

class SetCriterion(nn.Module):
    def __init__(self, matcher, args):
        super().__init__()
        self.phase = args.phase
        self.matcher = matcher
        self.dataset_name = args.dataset_name
        if self.phase == 0 or self.phase == 1:
            def network_loss(G, point_value, cw2, cw3):
                loss = 0
                loss_occ = dict()
                for key in G.keys():
                    loss_occ[key] = torch.mean((point_value - G[key]) ** 2)
                    loss += loss_occ[key]
                loss_weight = torch.sum(torch.abs(cw3 - 1)) + (
                    torch.sum(torch.clamp(cw2 - 1, min=0) - torch.clamp(cw2, max=0)))
                loss += loss_weight
                return loss, loss_occ
            
            self.loss = network_loss
        else:
            def network_loss(G, point_value, cw2, cw3):
                loss = 0
                loss_occ = dict()
                for key in G.keys():
                    loss_occ[key] = torch.mean(
                    (1 - point_value) * (1 - torch.clamp(G[key], max=1)) + point_value * (torch.clamp(G[key], min=0)))
                    loss += loss_occ[key]
                loss_weight = torch.sum((cw2 < 0.01).float() * torch.abs(cw2)) + torch.sum(
                    (cw2 >= 0.01).float() * torch.abs(cw2 - 1))
                loss += loss_weight
                return loss, loss_occ

            self.loss = network_loss



    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):

        match_indices = self.matcher(outputs, targets)

        if self.dataset_name == 'stru3d':
            pred_occ = {
                "shape_occ": outputs['pred_occ']
            }
        else:
            if self.phase == 0:
                pred_occ = {
                    "shape_occ": outputs['pred_occ']
                }
            else:
                pred_occ = {
                    "shape_occ": outputs['pred_occ'],
                    "axis_occ": outputs['axis_occ'],
                    "non_axis_occ": outputs['non_axis_occ'],
                }

        idx = self._get_src_permutation_idx(match_indices)
        target_occ_o = torch.cat([t[J] for t, (_, J) in zip(targets['occ'], match_indices)])
        target_occ = torch.full(pred_occ['shape_occ'].shape, 0, dtype=torch.float32, device=pred_occ['shape_occ'].device)
        target_occ[idx] = target_occ_o

        loss_value, loss_occ = self.loss(pred_occ, target_occ, outputs['binary_weights'], outputs['merge_weights'])

        loss = dict()
        loss['loss_occ'] = loss_occ
        loss['loss'] = loss_value

        return loss

def build(args):
    # define the model
    model = FRINet(args)
    
    # define the matcher
    matcher = build_matcher(args)
    
    # define the loss function
    criterion = SetCriterion(matcher, args)

    return model, criterion