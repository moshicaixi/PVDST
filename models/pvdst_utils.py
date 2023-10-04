import functools

import torch
import torch.nn as nn
import numpy as np

from modules.shared_mlp import SharedMLP
from modules.pointnet import PointNetSAModule, PointNetAModule, PointNetFPModule
from modules.se import SE3d
from modules.voxelization import Voxelization
import modules.functional as MF
from modules.ops import sample_and_group, sample_and_group_with_idx


class GeometryEncoding(nn.Module):
    def __init__(self, enc_channels, rel=True, abs=False, euc=True):
        """
        enc_channels: output_channels
        rel: relative position
        abs: absolute position
        euc: euclidean distance
        """
        super().__init__()
        self.rel = rel
        self.abs = abs
        self.euc = euc 
        
        in_channels = 0
        in_channels += 3 if rel else 0
        in_channels += 3 if abs else 0
        in_channels += 1 if euc else 0
        self.ge = nn.Sequential(
            nn.Conv2d(in_channels, enc_channels, 1), 
            nn.BatchNorm2d(enc_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(enc_channels, enc_channels, 1)
        )

    def forward(self, neighboe_xyz, center_xyz):
        xyz_diff = neighboe_xyz - center_xyz.unsqueeze(-1)

        enc_features_list = []
        if self.rel:
            enc_features_list.append(xyz_diff)
        if self.abs:
            enc_features_list.append(neighboe_xyz)
        if self.euc:
            enc_features_list.append(torch.norm(xyz_diff, p=2, dim=1).unsqueeze(1))

        enc_features = torch.cat(enc_features_list, dim=1)

        return self.ge(enc_features)


class NeighborEmbedding(nn.Module):
    def __init__(self, in_channels, emb_channels, form='squ', geo=True):
        """
        emb_channels: output_channels
        form: 'abs', 'squ', 'mul'
            abs: absolute difference
            squ: square difference
            mul: multiplication
        geo: geometry enconde
        """
        super().__init__()
        self.form = form
        self.geo = geo
        
        self.ne = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, emb_channels, 1),
        )

    def forward(self, key, query, geo=None):
        if self.form == 'abs':
            diff = key - query.unsqueeze(-1)
            emb_features = - torch.abs(diff)
        elif self.form == 'squ':
            diff = key - query.unsqueeze(-1)
            emb_features = - diff * diff
        elif self.form == 'mul':
            prod = key * query.unsqueeze(-1)
            emb_features = prod
        
        if self.geo:
            emb_features += geo

        return self.ne(emb_features)


class LAAttention(nn.Module):
    """
    Local Relation Attention
    """
    radius = 0.1
    def __init__(self, channels, shrink_ratio=4, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16}
                 ):
        super().__init__()
        self.v_channels = channels
        self.qk_channels = channels // shrink_ratio
        self.ratio = shrink_ratio
        
        self.grouper = group_args.get('NAME', 'ballquery')
        self.radius = group_args.get('radius', 0.1)
        self.nsample = group_args.get('nsample', 16)
        
        self.q_conv = nn.Conv1d(channels, self.qk_channels, 1)
        self.k_conv = nn.Conv1d(channels, self.qk_channels, 1)
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.geometry_encoding = GeometryEncoding(self.qk_channels, rel=True, abs=False, euc=True)
        self.neighbor_embedding = NeighborEmbedding(in_channels=self.qk_channels, emb_channels=channels, form='squ', geo=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, y_xyz=None, idx=None):
        """
        x: query vector
        y: key & value vector
        y_xyz: key & value vector's coordinates
        if x == y: self attention
        if x != y: cross attention
        """
        if y_xyz is None:
            raise ValueError("vector attention needs coordinates.")
        n = y.size(-1)
        x_q = self.q_conv(x)  # b, c/r, n
        y_k = self.k_conv(y)  # b, c/r, n
        y_v = self.v_conv(y)  # b, c, n
        if idx is None:
            y_k, grouped_xyz, new_xyz = sample_and_group(n, self.nsample, y_xyz, y_k, 
                                                         radius=self.radius, grouper=self.grouper)  # b, c/r+3, n, k
            y_v, _, _ = sample_and_group(n, self.nsample, y_xyz, y_v,
                                         radius=self.radius, grouper=self.grouper)  # b, c, n, k
        else:
            y_k, grouped_xyz, new_xyz = sample_and_group_with_idx(n, idx, y_xyz, y_k,
                                                                  radius=self.radius, grouper=self.grouper)  # b, c/r+3, n, k
            y_v, _, _ = sample_and_group_with_idx(n, idx, y_xyz, y_v,
                                                  radius=self.radius, grouper=self.grouper)  # b, c, n, k

        h_ij = self.geometry_encoding(grouped_xyz, new_xyz)
        w = self.neighbor_embedding(y_k, x_q, h_ij)

        w = self.softmax(w) 
        x = (y_v * w).sum(-1)
        return x


class PVDSA(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'bias_3d': True, 'normalize': False, 
                              'eps': 0, 'with_se': False, 'agg_way': 'add'}
                 ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        self.agg_way = aggr_args.get('agg_way', 'add')

        attention = LAAttention

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)
        self.v_cross_att = attention(channels, group_args=group_args)
        self.p_self_att = attention(channels, group_args=group_args)
        self.p_cross_att = attention(channels, group_args=group_args)

    def forward(self, inputs):
        x, xyz, idx = inputs

        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n
        p_p_x = self.p_self_att(x, x, xyz, idx) + x
        p_v_x2 = self.v_cross_att(p_v_x, p_p_x, xyz, idx) + p_v_x
        p_p_x2 = self.p_cross_att(p_p_x, p_v_x, xyz, idx) + p_p_x
        
        if self.agg_way == 'add':
            f_x = p_v_x2 + p_p_x2
        else:
            f_x = torch.cat((p_v_x2, p_p_x2), dim=1)

        return f_x


class PVDSA2(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'bias_3d': True, 'normalize': False, 
                              'eps': 0, 'with_se': False, 'agg_way': 'add'}
                 ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        self.agg_way = aggr_args.get('agg_way', 'add')

        attention = LAAttention

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)
        self.v_cross_att = attention(channels, group_args=group_args)
        self.p_self_att = attention(channels, group_args=group_args)
        self.p_cross_att = attention(channels, group_args=group_args)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.bn3 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x, xyz, idx = inputs

        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n
        p_p_x = self.act(self.bn1(self.p_self_att(x, x, xyz, idx))) + x
        p_v_x2 = self.act(self.bn2(self.v_cross_att(p_v_x, p_p_x, xyz, idx))) + p_v_x
        p_p_x2 = self.act(self.bn3(self.p_cross_att(p_p_x, p_v_x, xyz, idx))) + p_p_x
        
        if self.agg_way == 'add':
            f_x = p_v_x2 + p_p_x2
        else:
            f_x = torch.cat((p_v_x2, p_p_x2), dim=1)

        return f_x


class PVDSA3(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'bias_3d': True, 'normalize': False, 
                              'eps': 0, 'with_se': False, 'agg_way': 'add'}
                 ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        self.agg_way = aggr_args.get('agg_way', 'add')

        attention = LAAttention

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)
        self.v_cross_att = attention(channels, group_args=group_args)
        self.p_self_att = attention(channels, group_args=group_args)
        self.p_cross_att = attention(channels, group_args=group_args)

        self.mlp1 = nn.Conv1d(channels, channels, 1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.mlp2 = nn.Conv1d(channels, channels, 1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.mlp3 = nn.Conv1d(channels, channels, 1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x, xyz, idx = inputs

        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n
        p_p_x = self.act(self.bn1(self.mlp1(self.p_self_att(x, x, xyz, idx)))) + x
        p_v_x2 = self.act(self.bn2(self.mlp2(self.v_cross_att(p_v_x, p_p_x, xyz, idx)))) + p_v_x
        p_p_x2 = self.act(self.bn3(self.mlp3(self.p_cross_att(p_p_x, p_v_x, xyz, idx)))) + p_p_x
        
        if self.agg_way == 'add':
            f_x = p_v_x2 + p_p_x2
        else:
            f_x = torch.cat((p_v_x2, p_p_x2), dim=1)

        return f_x


class PVB(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'groups': 0, 'bias_3d': True, 
                              'normalize': False, 'eps': 0, 'with_se': False,
                              'proj_channel': 3, 'refine_way': 'cat', 'agg_way': 'add'}
                 ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        # grouper args
        self.grouper = group_args.get('NAME', 'ballquery')
        self.radius = group_args.get('radius', 0.1)
        self.nsample = group_args.get('nsample', 16)

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        groups = aggr_args.get('groups', 0)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        proj_channel = aggr_args.get('proj_channel', 3)
        self.refine_way = aggr_args.get('refine_way', 'cat')
        self.agg_way = aggr_args.get('agg_way', 'add')

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)

        self.proj_conv = nn.Sequential(
            nn.Conv2d(channels, proj_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channel),
            nn.ReLU(True),
            nn.Conv2d(proj_channel, proj_channel, kernel_size=1, bias=False)
        )
        self.proj_transform = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(True)
        )
        self.lift_conv = nn.Sequential(
            nn.BatchNorm2d(proj_channel*2),
            nn.ReLU(True),
            nn.Conv2d(proj_channel*2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        # self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

        self.softmax = nn.Softmax(dim=-1)
        self.beta = nn.Parameter(torch.ones([1]))

    def repeat_(self, x, part):
        repeat_dim = self.channels // part
        repeats = []
        for i in range(part-1):
            repeats.append(repeat_dim)
        repeats.append(self.channels-repeat_dim*(part-1))
        repeat_tensor = torch.tensor(repeats, dtype=torch.long, device=x.device, requires_grad=False)
        return torch.repeat_interleave(x, repeat_tensor, dim=1)

    def forward(self, inputs):
        x, xyz, idx = inputs
        n = x.size(-1)

        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n

        if idx is None:
            x_j, xyz_j, xyz_i = sample_and_group(n, self.nsample, xyz, x, grouper=self.grouper)  # b, c/r+3, n, k
        else:
            x_j, xyz_j, xyz_i = sample_and_group_with_idx(n, idx, xyz, x)  # b, c/r+3, n, k
        dp1 = xyz_j - xyz_i.unsqueeze(-1)  # b, 3, n, k
        df = x_j - x.unsqueeze(-1) # b, c, n, k
        dp2 = self.proj_conv(df)  # b, 3, n, k
        dp = torch.cat((dp1, dp2), dim=1)   # b, 3*2, n, k
        # dp = dp1 + dp2
        w = self.repeat_(dp, 6)  # b, c, n, k
        # w = self.lift_conv(dp)
        x_j = x_j * w
        x_j = self.proj_transform(x_j)
        p_p_x = self.pool(x_j)

        if self.agg_way == 'add':
            f_x = p_v_x + p_p_x
            # f_x = self.beta * p_v_x + (1 - self.beta) * p_p_x
        else:
            f_x = torch.cat((p_v_x, p_p_x), dim=1)

        return f_x


class PVB2(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'groups': 0, 'bias_3d': True, 
                              'normalize': False, 'eps': 0, 'with_se': False,
                              'proj_channel': 3, 'refine_way': 'cat', 'agg_way': 'add'}
                ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        # grouper args
        self.grouper = group_args.get('NAME', 'ballquery')
        self.radius = group_args.get('radius', 0.1)
        self.nsample = group_args.get('nsample', 16)

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        groups = aggr_args.get('groups', 0)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        proj_channel = aggr_args.get('proj_channel', 3)
        self.refine_way = aggr_args.get('refine_way', 'cat')
        self.agg_way = aggr_args.get('agg_way', 'add')

        self.shared = 6 if self.refine_way == 'cat' else 3
        mid_channel = int(np.ceil(channels / self.shared))

        # voxel depth-wise convolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)

        # point position-adaptive abstraction
        self.proj_conv = nn.Sequential(
            nn.Conv2d(channels, proj_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channel, proj_channel, kernel_size=1, bias=False)
        )
        self.pre_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        self.pos_conv = nn.Sequential(
            nn.Conv1d(mid_channel*self.shared, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.proj_transform = nn.Sequential(
            nn.BatchNorm2d(mid_channel*self.shared),
            nn.ReLU(inplace=True),
        )
        self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        # self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

        # self.beta = nn.Parameter(torch.ones([1]))        
        
    def forward(self, inputs):
        x, xyz, idx = inputs

        # 1. voxel convolution
        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n

        # neighbor grouping
        n = x.size(-1)
        if idx is None:
            x_j, _, _ = sample_and_group(n, self.nsample, xyz, x, radius=self.radius, grouper=self.grouper, use_xyz=True)  # b, c/r+3, n, k
        else:
            x_j, _, _ = sample_and_group_with_idx(n, idx, xyz, x, radius=self.radius, grouper=self.grouper, use_xyz=True)  # b, c/r+3, n, k
        
        # position-adaptive weight
        dp = x_j[:, 0:3, :, :]   # b, 3, n, k
        x_j = x_j[:, 3:, :, :]   # b, c, n, k
        df = x_j - x.unsqueeze(-1)  # b, c, n, k
        weight = self.proj_conv(df)
        if self.refine_way == 'cat':
            weight = torch.cat((dp, weight), dim=1)  # b, 3*2, n, k
        else:
            weight = dp + weight
            # weight = dp 

        # 2. point abstraction
        B, _, N, K = x_j.size()
        x_j = self.pre_conv(x_j)   # b, c/s, n, k
        x_j = x_j.unsqueeze(1).repeat(1, self.shared, 1, 1, 1) * weight.unsqueeze(2)   # b, s, c/s, n, k
        x_j = x_j.view(B, -1, N, K)   # b, c, n, k 
        # fj = self.proj_transform(fj)
        x = self.pool(x_j)   # b, c, n
        p_p_x = self.pos_conv(x)

        # 3. feature fusion
        if self.agg_way == 'add':
            f_x = p_v_x + p_p_x
            # f_x = self.beta * p_v_x + (1 - self.beta) * p_p_x
        else:
            f_x = torch.cat((p_v_x, p_p_x), dim=1)

        return f_x


class PVB3(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'groups': 0, 'bias_3d': True, 
                              'normalize': False, 'eps': 0, 'with_se': False,
                              'proj_channel': 3, 'refine_way': 'cat', 'agg_way': 'add'}
                ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        # grouper args
        self.grouper = group_args.get('NAME', 'ballquery')
        self.radius = group_args.get('radius', 0.1)
        self.nsample = group_args.get('nsample', 16)

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        groups = aggr_args.get('groups', 0)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        proj_channel = aggr_args.get('proj_channel', 3)
        self.refine_way = aggr_args.get('refine_way', 'cat')
        self.agg_way = aggr_args.get('agg_way', 'add')

        self.shared = 6 if self.refine_way == 'cat' else 3
        mid_channel = int(np.ceil(channels / self.shared))

        # voxel depth-wise convolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)

        # point position-adaptive abstraction
        self.proj_conv = nn.Sequential(
            nn.Conv2d(channels, proj_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channel, proj_channel, kernel_size=1, bias=False)
        )
        self.pre_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        self.pos_conv = nn.Sequential(
            nn.Conv1d(mid_channel*self.shared, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.proj_transform = nn.Sequential(
            nn.BatchNorm2d(mid_channel*self.shared),
            nn.ReLU(inplace=True),
        )
        self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        # self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
        
        # self.alpha = nn.Parameter(torch.ones([1], dtype=torch.float32))       
        self.beta = nn.Parameter(torch.ones([1], dtype=torch.float32))       
        
    def forward(self, inputs):
        x, xyz, idx = inputs

        # 1. voxel convolution
        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n

        # neighbor grouping
        n = x.size(-1)
        if idx is None:
            x_j, _, _ = sample_and_group(n, self.nsample, xyz, x, radius=self.radius, grouper=self.grouper, use_xyz=True)  # b, c/r+3, n, k
        else:
            x_j, _, _ = sample_and_group_with_idx(n, idx, xyz, x, radius=self.radius, grouper=self.grouper, use_xyz=True)  # b, c/r+3, n, k
        
        # position-adaptive weight
        dp = x_j[:, 0:3, :, :]   # b, 3, n, k
        x_j = x_j[:, 3:, :, :]   # b, c, n, k
        df = x_j - x.unsqueeze(-1)  # b, c, n, k
        weight = self.proj_conv(df)
        if self.refine_way == 'cat':
            weight = torch.cat((dp, weight), dim=1)  # b, 3*2, n, k
        else:
            weight = dp + weight

        # 2. point abstraction
        B, _, N, K = x_j.size()
        x_j = self.pre_conv(x_j)   # b, c/s, n, k
        x_j = x_j.unsqueeze(1).repeat(1, self.shared, 1, 1, 1) * weight.unsqueeze(2)   # b, s, c/s, n, k
        x_j = x_j.view(B, -1, N, K)   # b, c, n, k 
        # fj = self.proj_transform(fj)
        x = self.pool(x_j)   # b, c, n
        p_p_x = self.pos_conv(x)

        # 3. feature fusion
        if self.agg_way == 'add':
            # f_x = p_v_x + p_p_x
            # f_x = self.alpha * p_v_x + self.beta * p_p_x
            f_x = self.beta * p_v_x + (1 - self.beta) * p_p_x
        else:
            f_x = torch.cat((p_v_x, p_p_x), dim=1)

        return f_x


class PVDSA_Res(nn.Module):
    """
    Point Transformer Layer of PT2(Hengshaung).
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, resolution=16, pvdsa_class=1,
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'groups': 0, 'bias_3d': True, 
                              'normalize': False, 'eps': 0, 'with_se': False,
                              'proj_channel': 3, 'refine_way': 'cat', 'agg_way': 'add',  
                              'res': True, 'conv_res': False},
                 **kwargs
                 ):
        super().__init__()
        self.res = aggr_args.pop('res', True)
        self.conv_res = aggr_args.pop('conv_res', False)
        agg_way = aggr_args.get('agg_way', 'add')
        self.mid_channels = out_channels // 2

        self.mlp1 = nn.Conv1d(in_channels, self.mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.mid_channels)
        self.mlp2 = nn.Conv1d(self.mid_channels if agg_way=='add' else self.mid_channels*2, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.mlp3 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

        if pvdsa_class == 1:
            pvdsa_block = PVDSA 
        elif pvdsa_class == 2:
            pvdsa_block = PVDSA2 
        elif pvdsa_class == 3:
            pvdsa_block = PVDSA3 
        elif pvdsa_class == 4:
            pvdsa_block = PVB
        elif pvdsa_class == 5:
            pvdsa_block = PVB2
        elif pvdsa_class == 6:
            pvdsa_block = PVB3
        else:
            raise NotImplementedError(f'pvdsa_class {pvdsa_class} in PVDSA_Res is not implemented')
        self.pvdsa = pvdsa_block(self.mid_channels, resolution=resolution, group_args=group_args, aggr_args=aggr_args)

    def forward(self, inputs):
        x, xyz, idx = inputs
        p_x = self.act(self.bn1(self.mlp1(x)))  # b, c/2, n

        f_x = self.pvdsa((p_x, xyz, idx))

        if self.res:
            if self.conv_res:
                f_x = self.act(self.bn2(self.mlp2(f_x)) + self.bn3(self.mlp3(x)))
            else:
                f_x = self.act(self.bn2(self.mlp2(f_x)) + x)
        else:
            f_x = self.act(self.bn2(self.mlp2(f_x)))

        return (f_x, xyz, idx)


def _linear_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks, in_channels, nsample=16, with_se=False, normalize=True, eps=0, agg_way='add',
                               res=True, conv_res=True, width_multiplier=1, voxel_resolution_multiplier=1, pvdsa_class=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = int(r * out_channels)
        if voxel_resolution is None:
            block = SharedMLP
        else:
            block = functools.partial(PVDSA_Res, nsample=nsample, resolution=int(vr * voxel_resolution), with_se=with_se, 
                                      normalize=normalize, eps=eps, agg_way=agg_way, res=res, conv_res=conv_res, pvdsa_class=pvdsa_class)
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
    return layers, in_channels, concat_channels


def create_pointnet2_sa_components(sa_blocks, in_channels, nsample=16, with_se=False, normalize=True, eps=0, agg_way='add',
                                   res=True, conv_res=True, width_multiplier=1, voxel_resolution_multiplier=1, pvdsa_class=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    # in_channels = extra_feature_channels + 3

    sa_layers, sa_in_channels = [], []
    for conv_configs, sa_configs in sa_blocks:
        sa_in_channels.append(in_channels)
        sa_blocks = nn.ModuleList()
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVDSA_Res, nsample=nsample, resolution=int(vr * voxel_resolution), with_se=with_se, 
                                          normalize=normalize, eps=eps, agg_way=agg_way, res=res, conv_res=conv_res, pvdsa_class=pvdsa_class)
            for _ in range(num_blocks):
                sa_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
            # extra_feature_channels = in_channels
        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                      num_neighbors=num_neighbors)
        sa_blocks.append(block(in_channels=in_channels, out_channels=out_channels,
                               include_coordinates=True))
        # in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        in_channels = out_channels[-1]
        sa_layers.append(sa_blocks)

    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, nsample=16, with_se=False, normalize=True, eps=0, agg_way='add',
                                res=True, conv_res=True, width_multiplier=1, voxel_resolution_multiplier=1, pvdsa_class=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = nn.ModuleList()
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx], out_channels=out_channels)
        )
        in_channels = out_channels[-1]
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVDSA_Res, nsample=nsample, resolution=int(vr * voxel_resolution), with_se=with_se, 
                                          normalize=normalize, eps=eps, agg_way=agg_way, res=res, conv_res=conv_res, pvdsa_class=pvdsa_class)
            for _ in range(num_blocks):
                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        fp_layers.append(fp_blocks)

    return fp_layers, in_channels
