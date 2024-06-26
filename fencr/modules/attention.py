# coding=utf-8
from typing import Any

import torch
import numpy as np


def single_query_att_func(q, k, v, valid=None, scale=None):
    """
    :param query: ? * l * a
    :param key: ? * l * a
    :param value: ? * l * v
    :param valid: ? * l
    :return: ? * v
    """
    att_v = (q * k).sum(dim=-1)  # ? * l
    if scale is not None:
        att_v = att_v * scale  # ? * l
    att_v = att_v - att_v.max(dim=-1, keepdim=True)[0]  # ? * l
    if valid is not None:
        att_v = att_v.masked_fill(valid.le(0), -np.inf)  # ? * l
    att_w = att_v.softmax(dim=-1)  # ? * l
    att_w = att_w.masked_fill(torch.isnan(att_w), 0)  # ? * l
    result = (att_w.unsqueeze(dim=-1) * v).sum(dim=-2)  # ? * v
    return result, att_w


class SingleQueryAtt(torch.nn.Module):
    def __init__(self, input_size, att_size, act_func=torch.nn.ReLU):
        super().__init__()
        self.input_size = input_size
        self.att_size = att_size
        self.act_func = act_func
        self.init_modules()

    def init_modules(self):
        self.attention_layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.att_size),
            self.act_func(),
            torch.nn.Linear(self.att_size, 1, bias=False)
        )

    def forward(self, v, valid=None, scale=None):
        """
        :param value: ? * l * v
        :param valid: ? * l
        :return: ? * v
        """
        return single_query_att_func(q=self.attention_layers(v), k=1, v=v, valid=valid, scale=scale)


class MultiQueryAtt(torch.nn.Module):
    def forward(self, q, k, v, valid=None, scale=None):
        """
        根据q和k的两两匹配程度，把k对应的v加权平均
        :param q: Queries张量，? * L_q * a
        :param k: Keys张量，? * L_----------------------------------------------------------------------k * a
        :param v: Values张量，? * L_k * V
        :param scale: 缩放因子，浮点标量
        :param valid: ? * L_q * L_k， mask`
        :return context: # ? * L_q * V
        """
        att_v = torch.matmul(q, k.transpose(-1, -2))  # ? * L_q * L_k
        if scale is not None:
            att_v = att_v * scale  # ? * L_q * L_k
        att_v = att_v - att_v.max(dim=-1, keepdim=True)[0]  # ? * L_q * L_k
        if valid is not None:
            att_v = att_v.masked_fill(valid.le(0), -np.inf)  # ? * L_q * L_k
        att_w = att_v.softmax(dim=-1)  # ? * L_q * L_k
        att_w = att_w.masked_fill(torch.isnan(att_w), 0)  # ? * L_q * L_k
        result = torch.matmul(att_w, v)  # ? * L_q * V
        return result, att_w


class SelfAtt(torch.nn.Module):
    def __init__(self, input_size, query_size=-1, key_size=-1, value_size=-1):
        super().__init__()

        self.input_size = input_size
        self.query_size = query_size if query_size != 0 else self.input_size
        self.key_size = key_size if key_size != 0 else self.input_size
        self.value_size = value_size if value_size != 0 else self.input_size
        assert self.query_size == self.key_size \
               or (self.query_size < 0 and self.key_size == self.input_size) \
               or (self.key_size < 0 and self.query_size == self.input_size)
        self.att_size = input_size
        if self.query_size > 0:
            self.att_size = self.query_size
        if self.key_size > 0:
            self.att_size = self.key_size

        self.init_modules()

    def init_modules(self):
        if self.query_size > 0:
            self.query_layer = torch.nn.Linear(self.input_size, self.att_size, bias=False)
        else:
            self.query_layer = None
        if self.key_size > 0:
            self.key_layer = torch.nn.Linear(self.input_size, self.att_size, bias=False)
        else:
            self.key_layer = None
        if self.value_size > 0:
            self.value_layer = torch.nn.Linear(self.input_size, self.value_size, bias=False)
        else:
            self.value_layer = None

    def forward(self, x, valid=None, scale=None, act_v=None):
        def transfer_if_valid_layer(layer):
            result = x
            if layer is not None:
                result = layer(x)
                if act_v is not None:
                    result = act_v(result)
            return result

        att_query = transfer_if_valid_layer(self.query_layer)  # ? * L * a
        att_key = transfer_if_valid_layer(self.key_layer)  # ? * L * a
        att_value = transfer_if_valid_layer(self.value_layer)  # ? * L * v
        return MultiQueryAtt()(q=att_query, k=att_key, v=att_value, scale=scale, valid=valid)


class MultiHeadSelfAtt(SelfAtt):
    def __init__(self, input_size, query_size=-1, key_size=-1, value_size=-1, head_num=1):
        self.head_num = head_num
        SelfAtt.__init__(self, input_size=input_size, query_size=query_size, key_size=key_size,
                         value_size=value_size)

    def init_modules(self):
        if self.query_size > 0:
            self.query_layer = torch.nn.Linear(self.input_size, self.att_size * self.head_num, bias=False)
        else:
            self.query_layer = None
        if self.key_size > 0:
            self.key_layer = torch.nn.Linear(self.input_size, self.att_size * self.head_num, bias=False)
        else:
            self.key_layer = None
        if self.value_size > 0:
            self.value_layer = torch.nn.Linear(self.input_size, self.value_size * self.head_num, bias=False)
        else:
            self.value_layer = None

    def forward(self, x, valid=None, scale=None, act_v=None):
        # x.shape ?*L*V
        head_x = torch.cat([x] * self.head_num, dim=-1)  # ? * L * (V*h)
        if valid is not None:
            valid = torch.cat([valid.unsqueeze(dim=-3)] * self.head_num, dim=-3)  # ? * h * L * L

        def transfer_if_valid_layer(layer, head_size):
            result = head_x
            if layer is not None:
                result = layer(x)
                if act_v is not None:
                    result = act_v(result)
            result = torch.stack(result.split(head_size, dim=-1), dim=-3)
            return result

        att_query = transfer_if_valid_layer(self.query_layer, self.att_size)  # ? * h * L * a
        att_key = transfer_if_valid_layer(self.key_layer, self.att_size)  # ? * h * L * a
        att_value = transfer_if_valid_layer(
            self.value_layer, self.value_size if self.value_size > 0 else self.input_size)  # ? * h * L * v
        return MultiQueryAtt()(q=att_query, k=att_key, v=att_value, scale=scale, valid=valid)  # ? * h * L * v
