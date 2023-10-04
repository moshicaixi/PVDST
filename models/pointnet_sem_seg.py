import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pvcnn_utils import create_pointnet_components, create_mlp_components

__all__ = ['PointNet']


class PointNet_semseg(nn.Module):
    blocks = ((64, 3, None), (128, 1, None), (1024, 1, None))

    def __init__(self, num_classes, extra_feature_channels=6, width_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3

        layers, channels_point, _ = create_pointnet_components(blocks=self.blocks, in_channels=self.in_channels,
                                                               width_multiplier=width_multiplier)
        self.point_features = nn.Sequential(*layers)

        layers, channels_cloud = create_mlp_components(in_channels=channels_point, out_channels=[256, 128],
                                                       classifier=False, dim=1, width_multiplier=width_multiplier)
        self.cloud_features = nn.Sequential(*layers)

        layers, _ = create_mlp_components(
            in_channels=(channels_point + channels_cloud), out_channels=[512, 256, 0.3, num_classes],
            classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['features']

        point_features = self.point_features(inputs)
        cloud_features = self.cloud_features(point_features.max(dim=-1, keepdim=False).values)
        features = torch.cat([point_features, cloud_features.unsqueeze(-1).repeat([1, 1, inputs.size(-1)])], dim=1)
        return self.classifier(features), None


class get_model(nn.Module):
    def __init__(self, num_class, extra_channel=6):
        super(get_model, self).__init__()
        # extra_channel = 6
        self.feat = PointNet_semseg(num_classes=num_class, extra_feature_channels=extra_channel)

    def forward(self, x):
        x, trans_feat = self.feat(x)
        x = x.transpose(1, 2).contiguous()  # b, n, c
        x = F.log_softmax(x, dim=-1)
        return x, trans_feat


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight) 

        return total_loss
