# coding=utf-8

import torch


class ColdSampling(torch.nn.Module):
    def forward(self, vectors, cs_ratio):
        """
        :param vectors: ? * v
        :param cs_ratio: 0 < cs_ratio < 1
        :return:
        """
        cs_p = torch.empty(vectors.size()[:-1], device=vectors.device).fill_(cs_ratio).unsqueeze(dim=-1)  # ? * 1
        drop_pos = torch.bernoulli(cs_p)  # ? * 1
        random_vectors = torch.empty(vectors.size(), device=vectors.device).normal_(0, 0.01)  # ? * v
        cs_vectors = random_vectors * drop_pos + vectors * (-drop_pos + 1)  # ? * v
        return cs_vectors, drop_pos
