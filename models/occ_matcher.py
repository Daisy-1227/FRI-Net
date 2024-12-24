# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    We do the matching in query point level
    """
    def __init__(self, phase):
        super().__init__()
        self.phase = phase
        if self.phase == 0 or self.phase == 1:
            def occ_loss(pred_occ, tgt_occ):
                # loss_occ = torch.mean((tgt_occ - pred_occ) ** 2)
                # return loss_occ
                diff_square = (pred_occ.unsqueeze(1) - tgt_occ.unsqueeze(0)) ** 2
                loss = torch.mean(diff_square, dim=[2, 3])
                return loss
            self.occ_loss = occ_loss
        else:
            def occ_loss(pred_occ, tgt_occ):
                clamp_pred_occ_0 = torch.clamp(pred_occ.unsqueeze(1), max=1)
                clamp_pred_occ_1 = torch.clamp(pred_occ.unsqueeze(1), min=0)

                clamp_tgt_occ = tgt_occ.unsqueeze(0)
                loss_tensor = (1 - clamp_tgt_occ) * (1 - clamp_pred_occ_0) + clamp_tgt_occ * clamp_pred_occ_1
                loss = torch.mean(loss_tensor, dim=[2, 3])
                return loss
            self.occ_loss = occ_loss

    def forward(self, outputs, targets):
        """
        Args:
            outputs: shape_occ: [bs, num_polys, num_query, 1]: 8 * 20 * 4096 * 1
            targets: target_occ: a list of target occ, len(target_occ) == bs, each element in the list: [num_room * 4096 * 1]
        Returns:
        """
        with torch.no_grad():

            pred_occ = outputs['pred_occ']
            bs, num_polys = pred_occ.shape[:2]

            src_pred_occ = pred_occ.flatten(0, 1)

            tgt_occ = torch.cat([v for v in targets['occ']])

            cost_occ = self.occ_loss(src_pred_occ, tgt_occ)

            C = cost_occ

            C = C.view(bs, num_polys, -1).cpu()
            sizes = [len(v) for v in targets['occ']]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(phase=args.phase)