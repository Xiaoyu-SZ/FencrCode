# coding=utf-8
import torch
import torch.nn.functional as F
import numpy as np


class SoftmaxRankLoss(torch.nn.Module):
    def forward(self, prediction, label, neg_thresh: int = 0, loss_sum: int = 1):
        pos_neg_tag = label[:, :1].gt(neg_thresh).float()  # B * 1
        pre_softmax = (prediction - prediction.max(dim=1,
                       keepdim=True)[0]).softmax(dim=1)  # B * (1+S)
        target_pre = pre_softmax[:, :1]  # B * 1
        loss = -(target_pre * pos_neg_tag + (1 - pos_neg_tag)
                 * (1 - target_pre)).log()  # B * 1
        if loss_sum == 1:
            return loss.sum()
        return loss.mean()


class BPRRankLoss(torch.nn.Module):
    def forward(self, prediction, label, neg_thresh: int = 0, loss_sum: int = 1):
        '''
        计算rank loss，类似BPR-max，参考论文:
        @inproceedings{hidasi2018recurrent,
          title={Recurrent neural networks with top-k gains for session-based recommendations},
          author={Hidasi, Bal{\'a}zs and Karatzoglou, Alexandros},
          booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
          pages={843--852},
          year={2018},
          organization={ACM}
        }
        :param prediction: 预测值 [B * (1+S)]
        :param label: 标签 [B * (1+S)]
        :param loss_sum: 1=sum, other= mean
        :return:
        '''
        pos_neg_tag = label[:, :1].gt(neg_thresh).float()  # B * 1
        observed, sample = prediction[:, :1], prediction[:, 1:]  # B * 1, B * S
        sample_softmax = sample * pos_neg_tag  # B * S
        sample_softmax = (sample_softmax - sample_softmax.max(dim=1,
                          keepdim=True)[0]).softmax(dim=1)  # B * S
        sample_sigmoid = (pos_neg_tag * (observed - sample)).sigmoid()  # B * S
        loss = -(sample_sigmoid * sample_softmax).sum(dim=1).log()  # B
        if loss_sum == 1:
            return loss.sum()
        return loss.mean()


class DebiasedRankLoss(torch.nn.Module):
    def forward(self, prediction, label, neg_thresh: int = 0, loss_sum: int = 1,
                tau: float = 1.0, tau_plus: float = 0.1):
        '''
        @inproceedings{DBLP:conf/nips/ChuangRL0J20,
          author    = {Ching{-}Yao Chuang and
                       Joshua Robinson and
                       Yen{-}Chen Lin and
                       Antonio Torralba and
                       Stefanie Jegelka},
          title     = {Debiased Contrastive Learning},
          booktitle = {NeurIPS},
          year      = {2020}
        }
        :param prediction: 预测值 [B * (1+S)]
        :param label: 标签 [B * (1+S)]
        :param loss_sum: 1=sum, other= mean
        :return:
        '''
        exp = (prediction - prediction.max(dim=1, keepdim=True)[0]).exp()
        pos_idx = label.gt(neg_thresh).float()  # B * (1+S)
        neg_idx = 1 - pos_idx  # B * (1+S)
        pos = (pos_idx * exp).sum(dim=-1)  # B
        neg = (neg_idx * exp).sum(dim=-1)  # B
        pos_n = pos_idx.sum(dim=-1)  # B
        neg_n = pos_idx.sum(dim=-1)  # B
        ng = (-neg_n * tau_plus * pos / pos_n + neg) / (1 - tau_plus)  # B

        loss = -(pos / (pos + torch.max(ng, neg_n * np.exp(-1.0 / tau)))).log()  # B
        if loss_sum == 1:
            return loss.sum()
        return loss.mean()
