import torch
import torch.nn.functional as F

import modules.functional as MF
from modules.functional.backend import _backend


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, C, N]
        dst: target points, [B, C, M]
    Output:
        dist: per-point square distance, [B, M, N]
    """
    b, _, n = src.size()
    _, _, m = dst.size()
    dist = -2 * torch.matmul(dst.permute(0, 2, 1), src)  # B, M, N
    dist += torch.sum(src ** 2, 1).view(b, 1, n)
    dist += torch.sum(dst ** 2, 1).view(b, m, 1)  # B, M, N
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, C, N]
        idx: sample index data, [B, M]
    Return:
        new_points:, indexed points data, [B, C, M]
    """
    new_points = MF.gather(points, idx)  # b, c, m
    return new_points



def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, 3, N]
        new_xyz: query points, [B, 3, M]
    Return:
        group_idx: grouped points index, [B, M, nsample]
    """
    group_idx = MF.ball_query(new_xyz, xyz, radius, nsample)  # b, m, k
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, C, N]
        new_xyz: query points, [B, C, M]
    Return:
        group_idx: grouped po ints index, [B, M, nsample]
    """
    sqrdists = square_distance(xyz, new_xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)  # b, m, k
    return group_idx.int()


def furthest_point_sample(xyz, ncenter):
    """
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance to the sampled point set
    :param xyz: coordinates of points, FloatTensor[B, 3, N]
    :param nsample: int, M
    :return:
      idx: indices of sampled centers coordinates,  FloatTensor[B, M]
    """
    xyz = xyz.contiguous()
    idx = _backend.furthest_point_sampling(xyz, ncenter)  # b, m
    return idx


def sample_and_group(ncenter, nsample, xyz, points=None, radius=0.1, grouper='knn', rel_x=False, use_xyz=False):
    """
    Input:
        ncenter: number of center points.
        radius: radius for ball query group.
        nsample: number of neighbor points for one center point.
        xyz: input points position data, [B, 3, N]
        points: input points feature data, [B, C, N]
    Return:
        new_xyz: sampled points position data, [B, 3, ncenter, nsample]
        new_points: sampled points feature data, [B, 3+C, ncenter, nsample]
    """
    
    b, _, n = xyz.size()
    m = ncenter
    xyz = xyz.contiguous()  # b, 3, n

    if ncenter != n:
        samp_idx = furthest_point_sample(xyz, ncenter) # b, m
        new_xyz = index_points(xyz, samp_idx)  # b, 3, m
        new_points = index_points(points, samp_idx)  # b, c, m
    else:
        new_xyz = xyz  # b, 3, n
        new_points = points   # b, c, m
    # elif sampler == 'random':
    #     samp_idx = random_point_sample(xyz, ncenter).long() # b, m
    if grouper == 'knn':
        idx = knn_point(nsample, xyz, new_xyz)  # b, m, k
    elif grouper == 'ballquery':
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = MF.grouping(xyz, idx)  # b, 3, m, k
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(-1)  # b, 3, m, k
    if grouper == 'ballquery':
        grouped_xyz_norm /= radius
    grouped_points = MF.grouping(points, idx) # b, c, m, k
    if rel_x:
        grouped_points_norm = grouped_points - new_points.unsqueeze(-1)  # b, c, m, k
        # (df, fi), used in DGCNN
        grouped_points = torch.cat((grouped_points_norm, new_points.unsqueeze(-1).repeat(1, 1, 1, nsample)), dim=1)  # (b, 2c, m, k)
    if use_xyz:
        # (dp, fj), used in PointNet++
        grouped_points = torch.cat((grouped_xyz_norm, grouped_points), dim=1)  # b, (c+3 or 2c+3), m, k 
    return grouped_points, grouped_xyz, new_xyz


def sample_and_group_with_idx(ncenter, idx, xyz, points=None, radius=0.1, grouper='knn', rel_x=False, use_xyz=False):
        """
        Input:
            ncenter: number of center points.
            radius: radius for ball query group.
            nsample: number of neighbor points for one center point.
            xyz: input points position data, [B, 3, N]
            points: input points feature data, [B, C, N]
        Return:
            new_xyz: sampled points position data, [B, 3, ncenter, nsample]
            new_points: sampled points feature data, [B, 3+C, ncenter, nsample]
        """
        
        b, _, n = xyz.size()
        m = ncenter
        xyz = xyz.contiguous()  # b, 3, n
        nsample = idx.size(-1)

        if ncenter != n:
            samp_idx = furthest_point_sample(xyz, ncenter) # b, m
            new_xyz = index_points(xyz, samp_idx)  # b, 3, m
            new_points = index_points(points, samp_idx)  # b, c, m
        else:
            new_xyz = xyz  # b, 3, n
            new_points = points   # b, c, m
        assert idx.size(1) == ncenter, 'number of the center points shoule be the same as indexs'
        grouped_xyz = MF.grouping(xyz, idx)  # b, 3, m, k
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(-1)  # b, 3, m, k
        if grouper == 'ballquery':
            grouped_xyz_norm /= radius
        grouped_points = MF.grouping(points, idx) # b, c, m, k
        if rel_x:
            grouped_points_norm = grouped_points - new_points.unsqueeze(-1)  # b, c, m, k
            grouped_points = torch.cat((grouped_points_norm, new_points.unsqueeze(-1).repeat(1, 1, 1, nsample)), dim=1)  # b, 2c, m, k
        if use_xyz:
            grouped_points = torch.cat((grouped_xyz_norm, grouped_points), dim=1)  # b, (c+3 or 2c+3), m, k 
        return grouped_points, grouped_xyz, new_xyz