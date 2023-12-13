# coding=utf-8

import torch
import numpy as np
from collections import defaultdict

from ..configs.constants import *
from ..metrics.MetricList import MetricsList
from ..metrics import rec_metrics as mmrec


class RecMetricsList(MetricsList):
    support_metrics = {
        **MetricsList.support_metrics,
        'hit@': mmrec.HitRatioAt,
        'ndcg@': mmrec.NDCGAt,
        'precision@': mmrec.PrecisionAt,
        'recall@': mmrec.RecallAt,
        'uauc': mmrec.UAUC,
        'auc': mmrec.AUROC,
    }

    def __init__(self, *args, **kwargs):
        self.require_rank = False
        super().__init__(*args, **kwargs)

    def parse_metrics_str(self, metrics_str: str):
        if type(metrics_str) is str:
            metrics_str = metrics_str.lower().strip().split(METRIC_SPLITTER)
        metrics = []
        for metric in metrics_str:
            metric = metric.strip()
            # uwarm, iwarm = None, None
            # if ':u' in metric:
            #     metric, uwarm = metric.split(':u')
            #     uwarm = ['-1'] + [i for i in uwarm.split(UWARM_SPLITTER)] + ['-1']
            if metric == '':
                continue
            tmp_metrics = []
            if '@' not in metric and metric in self.support_metrics:
                tmp_metrics.append(metric)
            else:
                metric, topk = metric.split('@')
                metric = metric + '@'
                if metric not in self.support_metrics:
                    continue
                topk = [k.strip() for k in topk.split(RANK_SPLITTER)]
                for k in topk:
                    tmp_metrics.append(metric + k)
            # if uwarm is not None:
            #     tmp = []
            #     for tmp_metric in tmp_metrics:
            #         tmp.append(tmp_metric)
            #         for lo, hi in zip(uwarm[:-1], uwarm[1:]):
            #             tmp.append(tmp_metric + ':u' + lo + '.' + hi)
            #     tmp_metrics = tmp
            metrics.extend(tmp_metrics)
        return metrics

    def init_metrics(self):
        metric_topks = defaultdict(list)
        for metric in self.metrics_str:
            metric = metric.strip()
            m_key = metric
            # uwarm, iwarm = None, None
            # if ':u' in metric:
            #     m_key, uwarm = metric.split(':u')
            #     uwarm = [int(i) for i in uwarm.split('~')]
            if '@' not in metric and metric not in self.metrics:
                metric_object = self.support_metrics[m_key](**self.metrics_kwargs)
                # if uwarm is not None:
                #     metric_object = mmrec.UserWarmMetric(
                #         min_cnt=None if uwarm[0] < 0 else uwarm[0], max_cnt=None if uwarm[1] < 0 else uwarm[1] - 1,
                #         metric_object=metric_object, **self.metrics_kwargs)
                self.metrics[metric] = metric_object
            elif '@' in m_key:
                rank_m, topk = m_key.split('@')
                rank_m = rank_m + '@'
                # if uwarm is not None:
                #     rank_m = (rank_m, uwarm[0], uwarm[1])
                topk = int(topk)
                if topk not in metric_topks[rank_m]:
                    metric_topks[rank_m].append(topk)
                    self.require_rank = True
        for metric in metric_topks:
            if type(metric) is str:
                self.metrics[metric] = self.support_metrics[metric](topks=metric_topks[metric], **self.metrics_kwargs)
            # else:
            #     m_key, lo, hi = metric
            #     metric_object = self.support_metrics[m_key](topks=metric_topks[metric], **self.metrics_kwargs)
            #     metric_object = mmrec.UserWarmMetric(
            #         min_cnt=None if lo < 0 else lo, max_cnt=None if hi < 0 else hi - 1,
            #         metric_object=metric_object, **self.metrics_kwargs)
            #     self.metrics[m_key + 'u:{}.{}'.format(lo, hi)] = metric_object

    def update(self, output: dict, ranked=False) -> None:
        prediction, label = output[PREDICTION], output[LABEL]
        prediction_at, label_at = None, None
        prediction_po, label_po = None, None
        for key in self.metrics:
            metric = self.metrics[key]
            if issubclass(type(metric), mmrec.RankMetric):
                if prediction_at is None:
                    if IID in output:
                        prediction_at = prediction.masked_fill(output[IID].le(0), -np.inf)
                    else:
                        prediction_at = prediction
                    if self.require_rank and not ranked:
                        prediction_at, label_at = mmrec.RankMetric.sort_rank(prediction_at, label)
                metric.update({**output, PREDICTION: prediction_at, LABEL: label_at}, ranked=True)
            else:
                if prediction_po is None:
                    if IID in output:
                        prediction_po, label_po = prediction.flatten(), label.flatten()
                        mask_idx = output[IID].flatten().gt(0).nonzero(as_tuple=True)[0]
                        prediction_po, label_po = prediction_po[mask_idx], label_po[mask_idx]
                    else:
                        prediction_po, label_po = prediction, label
                metric.update({**output, PREDICTION: prediction_po, LABEL: label_po})

    def compute(self, reset=False):
        result = {}
        for metric in self.metrics:
            metric_result = self.metrics[metric].compute()
            for k in metric_result:
                result[k] = metric_result[k]
        if reset:
            self.reset()
        return {k: result[k] for k in self.metrics_str}
