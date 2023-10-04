import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pvcnn_utils import create_pointnet_components
from models.pointnet_utils import feature_transform_reguliarzer
from modules.ops import knn_point, cal_loss

__all__ = ['PVCNN']


class PVCNN(nn.Module):
    blocks = ((64, 1, 32), (128, 2, 16), (512, 1, None), (2048, 1, None))

    def __init__(self, num_classes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3

        layers, channels_point, _ = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)

        # layers, _ = create_mlp_components(in_channels=(num_shapes + channels_point + concat_channels_point),
        #                                   out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
        #                                   classifier=True, dim=2, width_multiplier=width_multiplier)
        # self.classifier = nn.Sequential(*layers)
        channels_point = channels_point * width_multiplier
        # self.classifier = PCTClassifierCls2(channels_point, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
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
        features = inputs[:, :self.in_channels, :]
        coords = features[:, :3, :]

        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
        global_features = features.max(dim=-1).values

        return self.classifier(global_features), None


class get_model(nn.Module):
    def __init__(self, k=40, extra_channel=0):
        super(get_model, self).__init__()

        self.pvcnn = PVCNN(num_classes=k, extra_feature_channels=extra_channel)

    def forward(self, x):
        x, trans_feat = self.pvcnn(x)
        # x = F.log_softmax(x, dim=1)
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, smoothing=True):
        total_loss = cal_loss(pred, target)

        return total_loss


# class get_loss(torch.nn.Module):
#     def __init__(self, mat_diff_loss_scale=0.001):
#         super(get_loss, self).__init__()
#         self.mat_diff_loss_scale = mat_diff_loss_scale

#     def forward(self, pred, target, trans_feat):
#         loss = F.nll_loss(pred, target)
#         if trans_feat is None:
#             total_loss = loss
#         else:
#             mat_diff_loss = feature_transform_reguliarzer(trans_feat)
#             total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
#         return total_loss
