# coding=utf-8

from dis import disco
import torch
import torchmetrics
from torchmetrics.retrieval.retrieval_metric import RetrievalMetric
from torchmetrics.functional import auroc

from ..configs.constants import *
from ..metrics import metrics as mm


class RankMetric(torchmetrics.Metric):
    def __init__(self, topks: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(topks) is not list:
            topks = [topks]
        self.topks = topks
        for topk in topks:
            self.add_state(name='at{}'.format(topk),
                           default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state(name='total', default=torch.tensor(
            0.0), dist_reduce_fx='sum')

    @staticmethod
    def sort_rank(preds, target):
        assert preds.shape == target.shape
        shuffle_indices = torch.randperm(
            preds.size(-1), device=preds.device)  # n
        preds = preds[..., shuffle_indices]  # ? * n
        target = target[..., shuffle_indices]  # ? * n
        preds, indices = preds.sort(dim=-1, descending=True)  # ? * n
        target = target.gather(dim=-1, index=indices)  # ? * n
        return preds, target

    def update(self, output: dict, ranked: bool = False) -> None:
        preds, target = output[PREDICTION], output[LABEL]
        if not ranked:
            preds, target = self.sort_rank(preds, target)  # ? * n
        metrics = self.metric_at_k(preds, target)
        for topk in self.topks:
            metric = metrics[topk]
            exec("self.at{} += metric.sum()".format(topk))
        for topk in self.topks:
            self.total += metrics[topk].numel()
            break

    def compute(self):
        result = {}
        for topk in self.topks:
            result[topk] = eval("self.at{}".format(topk)) / self.total
        return result

    def metric_at_k(self, preds, target) -> dict:
        pass


class HitRatioAt(RankMetric):
    def metric_at_k(self, preds, target) -> dict:
        target = target.gt(0).float()  # ? * n
        result = {}
        for topk in self.topks:
            l = target[..., :topk] if topk > 0 else target  # ? * K
            hit = l.sum(dim=-1).gt(0).float()  # ?
            result[topk] = hit
        return result

    def compute(self):
        result = super().compute()
        return {'hit@{}'.format(topk): result[topk] for topk in self.topks}


class RecallAt(RankMetric):
    def metric_at_k(self, preds, target) -> dict:
        target = target.gt(0).float()  # ? * n
        total = target.sum(dim=-1)  # ?
        result = {}
        for topk in self.topks:
            l = target[..., :topk] if topk > 0 else target  # ? * K
            l = l.sum(dim=-1)  # ?
            recall = l / total  # ?
            result[topk] = recall
        return result

    def compute(self):
        result = super().compute()
        return {'recall@{}'.format(topk): result[topk] for topk in self.topks}


class PrecisionAt(RankMetric):
    def metric_at_k(self, preds, target) -> dict:
        target = target.gt(0).float()  # ? * n
        result = {}
        for topk in self.topks:
            l = target[..., :topk] if topk > 0 else target  # ? * K
            precision = l.mean(dim=-1)  # ?
            result[topk] = precision
        return result

    def compute(self):
        result = super().compute()
        return {'precision@{}'.format(topk): result[topk] for topk in self.topks}


class NDCGAt(RankMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('discounts', torch.log2(
            torch.arange(max(self.topks)) + 2.0))  # K

    def metric_at_k(self, preds, target) -> dict:
        ideal_rank, _ = target.sort(dim=-1, descending=True)  # ? * n
        result = {}
        for topk in self.topks:
            discount = self.discounts[:topk]
            l = target[..., :topk] if topk > 0 else target  # ? * K
            dcg = (l / discount).sum(dim=-1)  # ?
            ideal = ideal_rank[..., :topk] if topk > 0 else ideal_rank  # ? * K
            idcg = (ideal / discount).sum(dim=-1)  # ?
            ndcg = dcg / idcg  # ?
            result[topk] = ndcg
        return result

    def compute(self):
        result = super().compute()
        return {'ndcg@{}'.format(topk): result[topk] for topk in self.topks}


class AUROC(mm.AUROC):
    def update(self, output: dict):
        super().update(
            {PREDICTION: output[PREDICTION].sigmoid(), LABEL: output[LABEL].gt(0).long()})


class UAUC(RetrievalMetric):
    def update(self, output: dict):
        return super().update(preds=output[PREDICTION].sigmoid(), target=output[LABEL].gt(0).long(),
                              indexes=output[UID])

    def _metric(self, preds, target):
        return auroc(preds, target, pos_label=1)

    def compute(self) -> torch.Tensor:
        """
        First concat state `indexes`, `preds` and `target` since they were stored as lists. After that,
        compute list of groups that will help in keeping together predictions about the same query.
        Finally, for each group compute the `_metric` if the number of positive targets is at least
        1, otherwise behave as specified by `self.empty_target_action`.
        """
        indexes = torch.cat(self.indexes, dim=0)
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        res = []
        groups = get_group_indexes(indexes)

        for group in groups:
            mini_preds = preds[group]
            mini_target = target[group]

            if mini_target.min() == mini_target.max():
                continue
            else:
                # ensure list containt only float tensors
                res.append(self._metric(mini_preds, mini_target))
        return torch.stack([x.to(preds) for x in res]).mean() if len(res) > 0 \
            else torch.tensor(0.0, device=preds.device)


# class UserWarmMetric(torchmetrics.Metric):
#     def __init__(self, min_cnt, max_cnt, metric_object, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.min_cnt, self.max_cnt = min_cnt, max_cnt
#         self.metric_object = metric_object
#         assert self.min_cnt is not None or self.max_cnt is not None
#
#     def update(self, preds, target, user_cnt, *args, **kwargs) -> None:
#         if self.min_cnt is None:
#             cnt_filter = lambda x: x.le(self.max_cnt).byte()
#         elif self.max_cnt is None:
#             cnt_filter = lambda x: x.ge(self.min_cnt).byte()
#         else:
#             cnt_filter = lambda x: x.le(self.max_cnt).byte() * x.ge(self.min_cnt).byte()
#         idx = cnt_filter(user_cnt).nonzero(as_tuple=True)[0]
#         self.metric_object.update(preds=preds[idx], target=target[idx], *args, **kwargs)
#
#     def compute(self):
#         return self.metric_object.compute()
