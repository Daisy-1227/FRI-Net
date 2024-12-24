import torch
import torch.nn.functional as F
from torch import nn, Tensor

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class room_wise_decoder(nn.Module):
    def __init__(self, num_line, args):
        super().__init__()
        self.phase = args.phase
        self.hidden_dim = args.hidden_dim
        self.num_line = num_line
        self.num_convex = args.num_convex

        # Line Prediction
        self.num_horizontal_line = args.num_horizontal_line
        self.num_vertical_line = args.num_vertical_line
        self.num_diagnoal_line = args.num_diagnoal_line
        
        self.horizontal_mlp = MLP(input_dim=self.hidden_dim * 40, output_dim=self.num_horizontal_line * 2,
                                  hidden_dim=self.hidden_dim * 2, num_layers=3)
        self.vertical_mlp = MLP(input_dim=self.hidden_dim * 40, output_dim=self.num_vertical_line * 2,
                                hidden_dim=self.hidden_dim * 2, num_layers=3)
        self.diagonal_mlp =  MLP(input_dim=self.hidden_dim * 40, output_dim=self.num_diagnoal_line * 3,
                                 hidden_dim=self.hidden_dim * 2, num_layers=3)

        # Line Grouping
        if self.phase == 0 or self.phase == 1:
            binary_matrix = torch.zeros((self.num_horizontal_line + self.num_vertical_line, self.num_convex))
        else:
            binary_matrix = torch.zeros((self.num_horizontal_line + self.num_vertical_line + self.num_diagnoal_line, self.num_convex))

        self.binary_matrix = nn.Parameter(binary_matrix)
        
        # Shape Assembly    
        if self.phase == 0 or self.phase == 1:
            merge_matrix = torch.zeros((self.num_convex, 1))
        else:
            merge_matrix = torch.zeros((self.num_convex * 2, 1))
            
        self.merge_matrix = nn.Parameter(merge_matrix)

        # Weights Initialization
        nn.init.normal_(self.binary_matrix, mean=0.0, std=0.02)
        nn.init.normal_(self.merge_matrix, mean=1e-5, std=0.02)

    # def update_weights(self):
    #     diagonal_binary_matrix = torch.zeros((self.num_diagnoal_line, self.num_convex)).cuda()
    #     diagonal_binary_matrix = nn.Parameter(diagonal_binary_matrix)
    #     nn.init.normal_(diagonal_binary_matrix, mean=0.0, std=0.02)
    #     self.binary_matrix = nn.Parameter(torch.cat([self.binary_matrix, diagonal_binary_matrix]))

    def update_weights(self):
        diagonal_binary_matrix = torch.zeros((self.num_diagnoal_line, self.num_convex)).cuda()
        diagonal_binary_matrix = nn.Parameter(diagonal_binary_matrix)
        nn.init.normal_(diagonal_binary_matrix, mean=0.0, std=0.02)
        self.binary_matrix = nn.Parameter(torch.cat((self.binary_matrix, diagonal_binary_matrix)))

        diagonal_merge_matrix = torch.zeros((self.num_convex, 1)).cuda()
        diagonal_merge_matrix = nn.Parameter(diagonal_merge_matrix)
        nn.init.normal_(diagonal_merge_matrix, mean=0.0, std=0.02)
        self.merge_matrix = nn.Parameter(torch.cat((self.merge_matrix, diagonal_merge_matrix)))

    # Input: room latent code, query point
    # Output: learned lines, occupancy
    def forward(self, room_latent, query):

        # Input latent: [bs, num_room, num_query_per_room, hidden_dim]
        [bs, num_room, num_query_per_room, hidden_dim] = room_latent.shape
        room_latent = room_latent.reshape(bs * num_room, num_query_per_room, hidden_dim)
        num_query = query.size(2)
        query = query.reshape(bs * num_room, query.size(2), query.size(3))

        num_latent = room_latent.shape[0]

        # Line prediction
        horizontal_param_ = self.horizontal_mlp(room_latent.view(num_latent, -1)).view(num_latent, 2, self.num_horizontal_line)
        vertical_param_ = self.vertical_mlp(room_latent.view(num_latent, -1)).view(num_latent, 2, self.num_vertical_line)
        
        # horizontal_param_ = self.horizontal_mlp(room_latent.mean(axis=1)).view(num_latent, 2, self.num_horizontal_line)
        # vertical_param_ = self.vertical_mlp(room_latent.mean(axis=1)).view(num_latent, 2, self.num_vertical_line)
        
        horizontal_param = torch.zeros(num_latent, 3, self.num_horizontal_line).cuda()
        vertical_param = torch.zeros(num_latent, 3, self.num_vertical_line).cuda()
        horizontal_param[:, :1, :] = horizontal_param_[:, :1, :]
        horizontal_param[:, 2:, :] = horizontal_param_[:, 1:, :]
        vertical_param[:, 1:2, :] = vertical_param_[:, :1, :]
        vertical_param[:, 2:, :] = vertical_param_[:, 1:, :]
        
        if self.phase == 0:
            # Only focus on the horizontal line and vertical line
            line_param = torch.cat((horizontal_param, vertical_param), dim=2)
            diagnoal_param = None
        else:
            # line_param = torch.cat((horizontal_param, vertical_param), dim=2)
            diagnoal_param = self.diagonal_mlp(room_latent.view(num_latent, -1)).view(num_latent, 3, self.num_diagnoal_line)
            line_param = torch.cat([horizontal_param, vertical_param, diagnoal_param], dim=2)
        
        if self.phase == 0:
            # Compute inside or outside from the line
            h1 = torch.matmul(query, line_param)
            h1 = torch.clamp(h1, min=0)

            # Line grouping: Compute inside or outside from the convex
            h2 = torch.matmul(h1, self.binary_matrix)
            h2 = torch.clamp(1 - h2, min=0, max=1)

            # Shape assembly: Compute inside or outside from the room
            h3 = torch.matmul(h2, self.merge_matrix)
            h3 = torch.clamp(h3, min=0, max=1)
            axis_h3 = h3
            diagnoal_h3 = None
            axis_h3 = axis_h3.reshape(bs, num_room, num_query, 1)

        elif self.phase == 1:
            # Compute inside or outside from the line
            h1 = torch.matmul(query, line_param[:, :, :self.num_horizontal_line + self.num_vertical_line])
            h1 = torch.clamp(h1, min=0)
            h2 = torch.matmul(h1, self.binary_matrix[:self.num_horizontal_line+self.num_vertical_line, :])
            h2 = torch.clamp(1 - h2, min=0, max=1)

            axis_h3 = torch.matmul(h2, self.merge_matrix[:self.num_convex])
            axis_h3 = torch.clamp(axis_h3, min=0, max=1)

            diagonal_h1 = torch.matmul(query, diagnoal_param)
            diagonal_h1 = torch.clamp(diagonal_h1, min=0)
            diagonal_h2 = torch.matmul(diagonal_h1, self.binary_matrix[self.num_horizontal_line+self.num_vertical_line:, :])
            diagonal_h2 = torch.clamp(1 - diagonal_h2, min=0, max=1)

            diagnoal_h3 = torch.matmul(diagonal_h2, self.merge_matrix[self.num_convex:])
            diagnoal_h3 = torch.clamp(diagnoal_h3, min=0, max=1)

            h2 = torch.cat([h2, diagonal_h2], dim=2)

            h3 = torch.matmul(h2, self.merge_matrix)
            h3 = torch.clamp(h3, min=0, max=1)
            diagnoal_h3 = diagnoal_h3.reshape(bs, num_room, num_query, 1)
            axis_h3 = axis_h3.reshape(bs, num_room, num_query, 1)

        else:
            # Compute inside or outside from the line
            h1 = torch.matmul(query, line_param[:, :, :self.num_horizontal_line + self.num_vertical_line])
            h1 = torch.clamp(h1, min=0)
            h2 = torch.matmul(h1, self.binary_matrix[:self.num_horizontal_line+self.num_vertical_line, :])
            # h2 = torch.clamp(1 - h2, min=0, max=1)
            axis_h3 = torch.min(h2, dim=2, keepdim=True)[0]

            diagonal_h1 = torch.matmul(query, diagnoal_param)
            diagonal_h1 = torch.clamp(diagonal_h1, min=0)
            diagonal_h2 = torch.matmul(diagonal_h1, self.binary_matrix[self.num_horizontal_line+self.num_vertical_line:, :])
            # diagonal_h2 = torch.clamp(1 - diagonal_h2, min=0, max=1)
            diagnoal_h3 = torch.min(diagonal_h2, dim=2, keepdim=True)[0]


            h2 = torch.cat([h2, diagonal_h2], dim=2)

            # h3 = torch.matmul(h2, self.merge_matrix)
            # h3 = torch.clamp(h3, min=0, max=1)
            h3 = torch.min(h2, dim=2, keepdim=True)[0]
            diagnoal_h3 = diagnoal_h3.reshape(bs, num_room, num_query, 1)
            axis_h3 = axis_h3.reshape(bs, num_room, num_query, 1)


        line_param = line_param.reshape(bs, num_room, 3, self.num_line)
        h2 = h2.reshape(bs, num_room, num_query, -1)
        h3 = h3.reshape(bs, num_room, num_query, 1)

        outputs = dict()
        outputs['line_param'] = line_param
        outputs['convex_occ'] = h2
        outputs['pred_occ'] = h3
        outputs['axis_occ'] = axis_h3
        outputs['non_axis_occ'] = diagnoal_h3
        return outputs


            

