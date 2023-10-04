import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pvdst_utils import PVDSA_Res
from modules.ops import knn_point, cal_loss

class PVDST_cls(nn.Module):
    """
    Point-Voxel Dual Stream Transformer for classification.
    """
    resolution = [32, 16, 16]
    out_channels = [128, 128, 128]
    group_args = {'NAME': 'knn', 'radius': 0.1, 'nsample': 16}
    aggr_args = {'kernel_size': 3, 'groups': 0, 'bias_3d': True, 
                 'normalize': False, 'eps': 0, 'with_se': False,
                 'agg_way': 'add', 'res': True, 'conv_res': False, 
                 'proj_channel': 3, 'refine_way': 'cat'
                 }

    def __init__(self, num_classes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3

        resolution = [r * voxel_resolution_multiplier for r in self.resolution]
        out_channels = [oc * width_multiplier for oc in self.out_channels]
        
        embed_channels = out_channels[0]
        self.nblock = len(out_channels)
        self.nsample = self.group_args.get('nsample', 16)

        self.input_embedding = nn.Sequential(
            nn.Conv1d(self.in_channels, embed_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_channels),
            nn.ReLU(),
            nn.Conv1d(embed_channels, embed_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_channels),
            nn.ReLU(),
        )

        self.encoder = nn.ModuleList()
        in_channels = embed_channels
        for i in range(self.nblock):
            self.encoder.append(
                PVDSA_Res(
                    in_channels=in_channels, 
                    out_channels=out_channels[i],  
                    resolution=resolution[i], 
                    pvdsa_class=5,
                    group_args=self.group_args,
                    aggr_args=self.aggr_args,
                )
            )
            in_channels = out_channels[i]

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(sum(out_channels), 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        x = inputs[:, :self.in_channels, :]
        xyz = x[:, 0:3, :]
        b, _, n = x.size()

        idx = knn_point(self.nsample, xyz, xyz)

        # input embedding
        x = self.input_embedding(x)

        # feature encoding
        xs = []
        for i in range(self.nblock):
            x, _, _ = self.encoder[i]((x, xyz, idx))
            xs.append(x)

        x = torch.cat(xs, dim=1)
        x = self.conv_fuse(x)  # b, 1024, n

        x_max = x.max(dim=-1).values.view(b, -1)  # b, c

        x = self.classifier(x_max)  # b, 50, n
        return x, None

    
class get_model(nn.Module):
    def __init__(self, k=40, extra_channel=0):
        super(get_model, self).__init__()

        self.pvdsnet = PVDST_cls(num_classes=k, extra_feature_channels=extra_channel)

    def forward(self, x):
        x, trans_feat = self.pvdsnet(x)
        # x = F.log_softmax(x, dim=1)

        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, smoothing=True):
        total_loss = cal_loss(pred, target)

        return total_loss





        

    
