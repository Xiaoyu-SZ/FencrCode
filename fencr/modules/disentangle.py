# coding=utf-8
import torch


def centered_distance(x, eps=1e-6):
    """
    input a batch of sampled vectors
    :param x: ? * n * v
    :return: ? * n * n
    """
    x_square = x.square().sum(dim=-1, keepdim=True)  # ? * n * 1
    pairwise_distance = -2 * torch.matmul(x, x.transpose(-1, -2)) + x_square + x_square.transpose(-1, -2)  # ? * n * n
    pairwise_distance = (pairwise_distance + eps).sqrt()  # ? * n * n
    pairwise_distance = pairwise_distance.masked_fill(torch.isnan(pairwise_distance), 0)  # ? * n * n
    row_mean = pairwise_distance.mean(dim=-1, keepdim=True)  # ? * n * 1
    col_mean = pairwise_distance.mean(dim=-2, keepdim=True)  # ? * 1 * n
    all_mean = row_mean.mean(dim=-2, keepdim=True)  # ? * 1 * 1
    return pairwise_distance - row_mean - col_mean + all_mean  # ? * n * n


def distance_correlation(x, y, eps=1e-6):
    """
    input two batch of sampled vectors
    :param x: ? * n * v
    :param y: ? * n * v
    :return:
    """
    a = centered_distance(x, eps=eps)  # ? * n * n
    b = centered_distance(y, eps=eps)  # ? * n * n
    # ignore 1/n^2
    ab_cov_square = (a * b).sum(dim=(-1, -2))  # ?
    a_var = (a * a).sum(dim=(-1, -2))  # ?
    b_var = (b * b).sum(dim=(-1, -2))  # ?
    cor = ab_cov_square / ((a_var * b_var + eps).sqrt() + eps)  # ?
    # print(ab_cov_square, a_var, b_var, cor)
    cor = cor.masked_fill(torch.isnan(cor), 0)  # ?
    return cor
