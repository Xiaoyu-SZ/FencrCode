# coding=utf-8

import numpy as np

from ..configs.constants import *
from ..datasets.Dataset import Dataset
from ..utilities.formatter import pad_array
from ..utilities.logging import create_tqdm
from ..utilities.rec import sample_iids


class RecDataset(Dataset):
    def __init__(self, model, phase: int, buffer_ds: int = 0,
                 *args, **kwargs):
        self.model = model
        self.phase = phase
        self.buffer_ds = 0 if model.train_sample_n > 0 and phase == TRAIN_PHASE else buffer_ds
        self.buffer_train_iids = None
        Dataset.__init__(self, model=model, phase=phase, buffer_ds=self.buffer_ds, *args, **kwargs)

    def get_interaction(self, index_dict: dict, index: int) -> dict:
        for c in [LABEL, UID, IID, TIME]:
            if c in self.data:
                index_dict[c] = [self.data[c][index]]
        return index_dict

    def sample_train_iids(self, sample_n: int, sample_neg_p: float, item_p=None):
        uids = self.data[UID] if UID in self.data else None
        if uids is not None:
            train_pos_his = {uid: self.reader.user_data[HIS_POS_SEQ][uid] \
                [: self.reader.user_data[HIS_POS_TRAIN][uid]] for uid in range(1, self.reader.user_num)}
            iids = sample_iids(sample_n=sample_n, uids=uids, item_num=self.reader.item_num,
                               exclude_iids=train_pos_his, replace=False, verbose=True, item_p=item_p)
            if sample_neg_p > 0:
                train_neg_his = {uid: self.reader.user_data[HIS_NEG_SEQ][uid] \
                    [: self.reader.user_data[HIS_NEG_TRAIN][uid]] for uid in range(1, self.reader.user_num)}
                n = np.array([min(len(train_neg_his[uid]), sample_n) for uid in uids])
                m = np.random.binomial(n=n, p=sample_neg_p)
                for idx, choice_n in create_tqdm(enumerate(m), total=len(m), desc='sample_train_iids from neg'):
                    if choice_n > 0:
                        iids[idx][:choice_n] = np.random.choice(train_neg_his[uids[idx]], size=choice_n, replace=False)
                        # candidates = train_neg_his[uids[idx]]
                        # candidates_p = None if item_p is None else item_p[candidates]
                        # candidates_p = None if candidates_p is None else candidates_p / candidates_p.sum()
                        # iids[idx][:choice_n] = np.random.choice(
                        #     candidates, size=choice_n, replace=False, p=candidates_p)
        else:
            iids = np.random.choice(self.reader.item_num, size=(len(self), sample_n), replace=False, p=item_p)
        self.buffer_train_iids = iids
        return iids

    def sample_train_iids_cloze(self, sample_n: int, max_his: int, cloze_p: float, cloze_last_p: float):
        seq_lohi_buffer, mask_pos_buffer, neg_iids_buffer = [], [], []
        user_data = self.reader.user_data
        cloze_train_idx = self.reader.cloze_train_idx
        last_ps = np.random.rand(*cloze_train_idx.shape)
        for uid, last_p in create_tqdm(zip(cloze_train_idx, last_ps), total=len(last_ps)):
            train_max_len = user_data[HIS_POS_TRAIN][uid]
            hi = train_max_len if max_his < 0 or train_max_len <= max_his \
                else np.random.randint(max_his, train_max_len + 1)
            lo = 0 if max_his < 0 else max(0, hi - max_his)
            seq_lohi_buffer.append([lo, hi])
            user_seq = user_data[HIS_POS_SEQ][uid][lo:hi]
            if last_p < cloze_last_p and len(user_seq) > 0:
                mask_pos = np.array([len(user_seq) - 1])
            else:
                mask_pos, = np.where(np.random.rand(*user_seq.shape) < cloze_p)
            mask_iids = user_seq[mask_pos]
            exclude_iids = {mask_pos[i]: [mask_iids[i]] for i in range(len(mask_pos))}
            neg_iids = sample_iids(sample_n=sample_n, uids=mask_pos, item_num=self.reader.item_num,
                                   exclude_iids=exclude_iids, replace=False, verbose=False)
            mask_pos_buffer.append(mask_pos.astype(int))
            neg_iids_buffer.append(neg_iids.reshape(len(mask_pos), sample_n).astype(int))
        self.buffer_train_iids = (np.array(seq_lohi_buffer),
                                  np.array(mask_pos_buffer, dtype=object),
                                  np.array(neg_iids_buffer, dtype=object))

    def extend_train_iids(self, index_dict: dict, index: int, label: int = 0) -> dict:
        iids = self.buffer_train_iids[index]
        index_dict = self.index_extend_key(index_dict, key=IID, data=iids)
        index_dict = self.index_extend_key(index_dict, key=LABEL, data=np.array([label] * len(iids), dtype=int))
        return index_dict

    def extend_eval_iids(self, index_dict: dict, index: int, sample_n: int) -> dict:
        iids = self.data[EVAL_IIDS][index][:sample_n]
        iids = pad_array(iids, max_len=sample_n, v=0)
        index_dict = self.index_extend_key(index_dict, key=IID, data=iids)
        labels = np.zeros(len(iids), dtype=int) if EVAL_LABELS not in self.data \
            else self.data[EVAL_LABELS][index][:sample_n]
        labels = pad_array(labels, max_len=sample_n, v=0)
        index_dict = self.index_extend_key(index_dict, key=LABEL, data=labels)
        return index_dict

    def eval_all_iids(self, index_dict: dict, index: int, ignore_pos=True):
        iids = np.arange(self.reader.item_num)
        labels = np.zeros(len(iids), dtype=int)
        if UID in index_dict and ignore_pos:
            uid = index_dict[UID][0]
            pos_his = self.reader.user_data[HIS_POS_SEQ][uid]
            if self.phase == TRAIN_PHASE:
                pos_his = pos_his[:self.reader.user_data[HIS_POS_TRAIN][uid]]
            elif self.phase == VAL_PHASE:
                pos_his = pos_his[:self.reader.user_data[HIS_POS_VAL][uid]]
            iids[pos_his] = 0
            if LABEL in index_dict:
                iids[index_dict[IID]] = index_dict[IID]
            if EVAL_LABELS in self.data:
                iids[self.data[EVAL_IIDS][index]] = self.data[EVAL_IIDS][index]
        if LABEL in index_dict:
            labels[index_dict[IID]] = index_dict[LABEL]
        if EVAL_LABELS in self.data:
            labels[self.data[EVAL_IIDS][index]] = self.data[EVAL_LABELS][index]
        index_dict[IID] = iids
        index_dict[LABEL] = labels
        return index_dict
