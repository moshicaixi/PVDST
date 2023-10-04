import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pvdst_utils import PVDSA_Res
from modules.ops import knn_point


class PVDST_partseg(nn.Module):
    """
    Point-Voxel Dual Stream Transformer for part segmentation.
    """
    shape_label_channels = 64

    resolution = [32, 16, 16]
    out_channels = [128, 128, 128]
    nsample = 16
    customized_parameters = {'kernel_size': 3, 'groups': 0, 'bias_3d': True, 
                             'normalize': False, 'eps': 0, 'with_se': False,
                             'agg_way': 'add', 'res': True, 'conv_res': False, 
                             'proj_channel': 3, 'refine_way': 'cat', 'grouper': 'knn'
                             }

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        cps = self.customized_parameters

        resolution = [r * voxel_resolution_multiplier for r in self.resolution]
        out_channels = [oc * width_multiplier for oc in self.out_channels]
        
        embed_channels = out_channels[0]
        self.nblock = len(out_channels)

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
                    nsample=self.nsample, 
                    resolution=resolution[i], 
                    with_se=cps.pop('with_se', False), 
                    normalize=cps.pop('normalize', False), 
                    eps=cps.pop('eps', 0), 
                    agg_way=cps.pop('agg_way', 'add'), 
                    res=cps.pop('res', True), 
                    conv_res=cps.pop('conv_res', False),
                    pvdsa_class=5,
                    **cps
                )
            )
            in_channels = out_channels[i]

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(sum(out_channels), 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )

        self.label_conv = nn.Sequential(
            nn.Conv1d(num_shapes, self.shape_label_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.shape_label_channels),
            nn.LeakyReLU(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.Conv1d(1024 * 3 + self.shape_label_channels, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, points, label):
        # inputs : [B, in_channels + S, N]
        x = points[:, :self.in_channels, :]
        xyz = x[:, 0:3, :]
        b, _, n = x.size()
        cls_label_one_hot = label.transpose(1, 2).repeat(1, 1, n)

        idx = knn_point(16, xyz, xyz)

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
        x_avg = x.mean(dim=-1).view(b, -1)  # b, c

        x_max_features = x_max.unsqueeze(-1).repeat(1, 1, n)
        x_avg_features = x_avg.unsqueeze(-1).repeat(1, 1, n)
        cls_label_feature = self.label_conv(cls_label_one_hot)
        x_global_features = torch.cat((x_max_features, x_avg_features, cls_label_feature), 1)  # b, 1024*2+64, n
        x = torch.cat((x, x_global_features), dim=1)  # b, 1024*3+64, n

        x = self.classifier(x)  # b, 50, n
        return x, None

    
class get_model(nn.Module):
    def __init__(self, part_num=50, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            extra_channel = 3
        else:
            extra_channel = 0

        self.pvdsnet = PVDST_partseg(num_classes=part_num, num_shapes=16, extra_feature_channels=extra_channel)

    def forward(self, point_cloud, label):
        x, trans_feat = self.pvdsnet(point_cloud, label)
        x = F.log_softmax(x, dim=1)

        return x.transpose(1, 2).contiguous(), trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)
        
        return total_loss





        

    
