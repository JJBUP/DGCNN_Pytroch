#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch


def knn(x, k):
    """
    Args:
        x: x为输入点云特征[B,C,N]
        k:
    如何计算xyz的距离呢？
    distance = (x0-xi)^2 + (y0-yi)^2 + (z0-zi)^2
             = x0^2 + xi^2 + y0^2 + yi^2 + z0^2 + zi^2 - 2x0xi - 2y0yi - 2y0yi
    如何计算特征的距离呢？特征大小越相同，点之间靠的越近，类比xyz即可,我们吧xyz想象乘特征的三个维度即可计算特征的相似
    为了方便使用矩阵计算，我们按照拆解的形式计算特征距离！
    """

    # _2xixj: [B , N , N]
    _2xixj = 2 * torch.matmul(x.transpose(2, 1), x)
    # xx: [B , 1 , N]
    xixj = torch.sum(x ** 2, dim=1, keepdim=True)  # 计算所有点 特征的 平方和
    pairwise_distance = -(xixj - _2xixj + xixj.transpose(2, 1))
    # 说实话，这不是knn，knn，knn是距离整个点集合最近的点，这里是距离中心点最近的点，如此做法应该是简便计算吧
    # idx: [B, N, k]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    """
    B:batch,C:coordination,D:data,N:num_point,k:num_knn
    Args:
        x:  [B,C,N] 或 [B,D,N] 输入数据
        k:  int,knn个数
        idx:[B, N, k],代表去idx索引的数据,idx = None则计算KNN,否则使用idx
        dim9:bool,dim9表示输出有9个维度,即原始 xyzrgb+edge
    """

    B = x.size(0)
    N = x.size(2)

    # 1. 计算knn索引，若提供索引则不计算knn
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)  # 语义分割时特征维度为9，678为norm_xyz


    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N  # batch区分索引[B,1,1] (0,1N,2N...)
    idx = idx + idx_base  # [B,N,K]
    idx = idx.view(-1)  # [B*N*K]
    D = x.size(1)
    x = x.transpose(2, 1).contiguous()  # [B, D ,N] ->[B, N ,D]
    feature = x.view(B * N, -1)[idx, :]  # [B, N, D]  -> [B*N, D]
    feature = feature.view(B, N, k, D)  # [B*N*k, D]->[B, N, k, D]
    # 3.点特征(绝对特征)与边特征(相对特征)拼接(类比绝对坐标与相对坐标)。
    x = x.view(B, N, 1, D).repeat(1, 1, k, 1)  # [B, N, k, D]
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # [B, 2D, N, k]

    return feature  # [B, 2D, N, k]
