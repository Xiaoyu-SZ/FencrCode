# coding=utf-8
from collections import Counter

import numpy as np
import scipy.sparse as sp

from ..data_readers.DataReader import DataReader
from ..utilities.formatter import *
from ..utilities.io import *


class RecReader(DataReader):
    def read_user(self, filename: str, formatters: dict) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)

        self.user_data = df2dict(df, formatters=formatters) if df is not None else {
            UID: np.array([0])}
        assert UID in self.user_data and len(
            self.user_data[UID]) == self.user_data[UID][-1] + 1
        self.reader_logger.debug("user data keys {}:{}".format(
            len(self.user_data), list(self.user_data.keys())))
        self.user_num = len(self.user_data[UID])
        self.reader_logger.info("user_num = {}".format(self.user_num))
        return self.user_data

    def read_item(self, filename: str, formatters: dict) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.item_data = df2dict(df, formatters=formatters) if df is not None else {
            IID: np.array([0])}
        assert IID in self.item_data and len(
            self.item_data[IID]) == self.item_data[IID][-1] + 1
        self.reader_logger.debug("item data keys {}:{}".format(
            len(self.item_data), list(self.item_data.keys())))
        self.item_num = len(self.item_data[IID])
        self.reader_logger.info("item_num = {}".format(self.item_num))
        return self.item_data

    def read_val_iids(self, filename: str, formatters: dict, sample_n: int = None) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        val_iids = df2dict(df, formatters=formatters)
        for c in val_iids:
            c_data = val_iids[c]
            if c in self.val_data:
                c_data = np.concatenate([self.val_data[c], c_data], axis=1)
            self.val_data[c] = filter_seqs(c_data, max_len=sample_n, padding=0)
        self.reader_logger.info("validation eval_iids = {}".format(
            self.val_data[EVAL_IIDS].shape))
        self.reader_logger.info("validation sample_n = {}".format(
            sample_n if sample_n >= 0 else self.item_num))
        return val_iids

    def read_test_iids(self, filename: str, formatters: dict, sample_n: int = None) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        test_iids = df2dict(df, formatters=formatters)
        for c in test_iids:
            c_data = test_iids[c]
            if c in self.test_data:
                c_data = np.concatenate([self.test_data[c], c_data], axis=1)
            self.test_data[c] = filter_seqs(
                c_data, max_len=sample_n, padding=0)
        self.reader_logger.info("test eval_iids = {}".format(
            self.test_data[EVAL_IIDS].shape))
        self.reader_logger.info("test sample_n = {}".format(
            sample_n if sample_n >= 0 else self.item_num))
        return test_iids

    def prepare_user_features(self, include_uid: bool = False,
                              multihot_features: str = None, numeric_features: str = None):
        self.reader_logger.debug("prepare user features...")
        uids = self.user_data.pop(
            UID) if not include_uid else self.user_data[UID]

        mh_f_dict, base = self.multihot_features(
            data_dicts=self.user_data, combine=multihot_features,
            k_filter=lambda x: x.startswith(USER_F) and x.endswith(CAT_F)
        )
        self.reader_logger.debug(
            'user multihot features = {}'.format(mh_f_dict))
        self.user_multihot_f_num = len(mh_f_dict)
        self.user_multihot_f_dim = base
        self.reader_logger.info(
            'user_multihot_f_num = {}'.format(self.user_multihot_f_num))
        self.reader_logger.info(
            'user_multihot_f_dim = {}'.format(self.user_multihot_f_dim))

        nm_f_dict = self.numeric_features(
            data_dicts=self.user_data, combine=numeric_features,
            k_filter=lambda x: x.startswith(USER_F) and (
                x.endswith(INT_F) or x.endswith(FLOAT_F))
        )
        self.reader_logger.debug(
            'user numeric features = {}'.format(nm_f_dict))
        self.user_numeric_f_num = len(nm_f_dict)
        self.reader_logger.info(
            'user_numeric_f_num = {}'.format(self.user_numeric_f_num))

        self.user_data[UID] = uids
        return {**mh_f_dict, **nm_f_dict}

    def prepare_item_features(self, include_iid: bool = False,
                              multihot_features: str = None, numeric_features: str = None):
        self.reader_logger.debug("prepare item features...")
        iids = self.item_data.pop(
            IID) if not include_iid else self.item_data[IID]

        mh_f_dict, base = self.multihot_features(
            data_dicts=self.item_data, combine=multihot_features,
            k_filter=lambda x: x.startswith(ITEM_F) and x.endswith(CAT_F)
        )
        self.reader_logger.debug(
            'item multihot features = {}'.format(mh_f_dict))
        self.item_multihot_f_num = len(mh_f_dict)
        self.item_multihot_f_dim = base
        self.reader_logger.info(
            'item_multihot_f_num = {}'.format(self.item_multihot_f_num))
        self.reader_logger.info(
            'item_multihot_f_dim = {}'.format(self.item_multihot_f_dim))

        nm_f_dict = self.numeric_features(
            data_dicts=self.item_data, combine=numeric_features,
            k_filter=lambda x: x.startswith(ITEM_F) and (
                x.endswith(INT_F) or x.endswith(FLOAT_F))
        )
        self.reader_logger.debug(
            'item numeric features = {}'.format(nm_f_dict))
        self.item_numeric_f_num = len(nm_f_dict)
        self.reader_logger.info(
            'item_numeric_f_num = {}'.format(self.item_numeric_f_num))

        self.item_data[IID] = iids
        return {**mh_f_dict, **nm_f_dict}

    def prepare_ctxt_features(self, include_time: bool = False,
                              multihot_features: str = None, numeric_features: str = None):
        self.reader_logger.debug("prepare context features...")
        data_dicts = [d for d in [self.train_data,
                                  self.val_data, self.test_data] if d is not None]
        times = [d.pop(TIME, None) if not include_time or TIME not in d else d[TIME]
                 for d in data_dicts]

        mh_f_dict, base = self.multihot_features(
            data_dicts=data_dicts, combine=multihot_features,
            k_filter=lambda x: x.startswith(CTXT_F) and x.endswith(CAT_F)
        )
        self.reader_logger.debug(
            'context multihot features = {}'.format(mh_f_dict))
        self.ctxt_multihot_f_num = len(mh_f_dict)
        self.ctxt_multihot_f_dim = base
        self.reader_logger.info(
            'ctxt_multihot_f_num = {}'.format(self.ctxt_multihot_f_num))
        self.reader_logger.info(
            'ctxt_multihot_f_dim = {}'.format(self.ctxt_multihot_f_dim))

        nm_f_dict = self.numeric_features(
            data_dicts=data_dicts, combine=numeric_features,
            k_filter=lambda x: x.startswith(CTXT_F) and (
                x.endswith(INT_F) or x.endswith(FLOAT_F))
        )
        self.reader_logger.debug(
            'context numeric features = {}'.format(nm_f_dict))
        self.ctxt_numeric_f_num = len(nm_f_dict)
        self.reader_logger.info(
            'ctxt_numeric_f_num = {}'.format(self.ctxt_numeric_f_num))
        for i, d in enumerate(data_dicts):
            if times[i] is not None:
                d[TIME] = times[i]
        return {**mh_f_dict, **nm_f_dict}

    def generate_inter_seq_hi(self, save_dict, group_id, save_id, filter_col, col_filter):
        hi_list = []
        for gid, sid, f in zip(group_id, save_id, filter_col):
            if gid not in save_dict:
                save_dict[gid] = []
            hi_list.append(len(save_dict[gid]))
            if col_filter(f):
                save_dict[gid].append(sid)
        return np.array(hi_list, dtype=int)

    def prepare_user_pos_his(self, label_filter=lambda x: x > 0):
        if HIS_POS_SEQ in self.user_data:
            return None
        self.reader_logger.info('prepare user pos his...')
        user_dict = {}
        self.train_data[HIS_POS_IDX] = self.generate_inter_seq_hi(
            save_dict=user_dict, group_id=self.train_data[UID], save_id=self.train_data[IID],
            filter_col=self.train_data[LABEL], col_filter=label_filter)
        self.user_data[HIS_POS_TRAIN] = np.array(
            [len(user_dict[uid]) if uid in user_dict else 0 for uid in range(self.user_num)])
        self.val_data[HIS_POS_IDX] = self.generate_inter_seq_hi(
            save_dict=user_dict, group_id=self.val_data[UID], save_id=self.val_data[IID],
            filter_col=self.val_data[LABEL], col_filter=label_filter)
        self.user_data[HIS_POS_VAL] = np.array(
            [len(user_dict[uid]) if uid in user_dict else 0 for uid in range(self.user_num)])
        self.test_data[HIS_POS_IDX] = self.generate_inter_seq_hi(
            save_dict=user_dict, group_id=self.test_data[UID], save_id=self.test_data[IID],
            filter_col=self.test_data[LABEL], col_filter=label_filter)
        self.user_data[HIS_POS_SEQ] = np.array(
            [np.array(user_dict[uid], dtype=int) if uid in user_dict else np.array([], dtype=int)
             for uid in range(self.user_num)], dtype=object)
        return user_dict

    def prepare_user_neg_his(self, label_filter=lambda x: x <= 0):
        if HIS_NEG_SEQ in self.user_data:
            return None
        self.reader_logger.info('prepare user neg his...')
        user_dict = {}
        self.train_data[HIS_NEG_IDX] = self.generate_inter_seq_hi(
            save_dict=user_dict, group_id=self.train_data[UID], save_id=self.train_data[IID],
            filter_col=self.train_data[LABEL], col_filter=label_filter)
        self.user_data[HIS_NEG_TRAIN] = np.array(
            [len(user_dict[uid]) if uid in user_dict else 0 for uid in range(self.user_num)])
        self.val_data[HIS_NEG_IDX] = self.generate_inter_seq_hi(
            save_dict=user_dict, group_id=self.val_data[UID], save_id=self.val_data[IID],
            filter_col=self.val_data[LABEL], col_filter=label_filter)
        self.user_data[HIS_NEG_VAL] = np.array(
            [len(user_dict[uid]) if uid in user_dict else 0 for uid in range(self.user_num)])
        self.test_data[HIS_NEG_IDX] = self.generate_inter_seq_hi(
            save_dict=user_dict, group_id=self.test_data[UID], save_id=self.test_data[IID],
            filter_col=self.test_data[LABEL], col_filter=label_filter)
        self.user_data[HIS_NEG_SEQ] = np.array(
            [np.array(user_dict[uid], dtype=int) if uid in user_dict else np.array([], dtype=int)
             for uid in range(self.user_num)], dtype=object)
        return user_dict

    def prepare_user_posneg_his(self, label_pos=lambda x: x > 0):
        if HIS_POSNEG_SEQ in self.user_data:
            return None
        self.reader_logger.info('prepare user posneg his...')
        user_dict = {}

        def generate_inter_seq_hi(data):
            hi_list = []
            for uid, iid, label in zip(data[UID], data[IID], data[LABEL]):
                if uid not in user_dict:
                    user_dict[uid] = []
                hi_list.append(len(user_dict[uid]))
                user_dict[uid].append(iid if label_pos(label) else -iid)
            return np.array(hi_list, dtype=int)

        self.train_data[HIS_POSNEG_IDX] = generate_inter_seq_hi(
            self.train_data)
        self.user_data[HIS_POSNEG_TRAIN] = np.array(
            [len(user_dict[uid]) if uid in user_dict else 0 for uid in range(self.user_num)])
        self.val_data[HIS_POSNEG_IDX] = generate_inter_seq_hi(self.val_data)
        self.user_data[HIS_POSNEG_VAL] = np.array(
            [len(user_dict[uid]) if uid in user_dict else 0 for uid in range(self.user_num)])
        self.test_data[HIS_POSNEG_IDX] = generate_inter_seq_hi(self.test_data)
        self.user_data[HIS_POSNEG_SEQ] = np.array(
            [np.array(user_dict[uid], dtype=int) if uid in user_dict else np.array([], dtype=int)
             for uid in range(self.user_num)], dtype=object)
        return user_dict

    def prepare_item_pos_his(self, label_filter=lambda x: x > 0):
        if HIS_POS_SEQ in self.item_data:
            return None
        self.reader_logger.info('prepare item pos his...')
        item_dict = {}
        self.train_data[HIS_POS_IDX] = self.generate_inter_seq_hi(
            save_dict=item_dict, group_id=self.train_data[IID], save_id=self.train_data[UID],
            filter_col=self.train_data[LABEL], col_filter=label_filter)
        self.item_data[HIS_POS_TRAIN] = np.array(
            [len(item_dict[iid]) if iid in item_dict else 0 for iid in range(self.item_num)])
        self.val_data[HIS_POS_IDX] = self.generate_inter_seq_hi(
            save_dict=item_dict, group_id=self.val_data[IID], save_id=self.val_data[UID],
            filter_col=self.val_data[LABEL], col_filter=label_filter)
        self.item_data[HIS_POS_VAL] = np.array(
            [len(item_dict[iid]) if iid in item_dict else 0 for iid in range(self.item_num)])
        self.test_data[HIS_POS_IDX] = self.generate_inter_seq_hi(
            save_dict=item_dict, group_id=self.test_data[IID], save_id=self.test_data[UID],
            filter_col=self.test_data[LABEL], col_filter=label_filter)
        self.item_data[HIS_POS_SEQ] = np.array(
            [np.array(item_dict[iid], dtype=int) if iid in item_dict else np.array([], dtype=int)
             for iid in range(self.item_num)], dtype=object)
        return item_dict

    def prepare_item_neg_his(self, label_filter=lambda x: x <= 0):
        if HIS_NEG_SEQ in self.item_data:
            return None
        self.reader_logger.info('prepare item neg his...')
        item_dict = {}
        self.train_data[HIS_NEG_IDX] = self.generate_inter_seq_hi(
            save_dict=item_dict, group_id=self.train_data[IID], save_id=self.train_data[UID],
            filter_col=self.train_data[LABEL], col_filter=label_filter)
        self.item_data[HIS_NEG_TRAIN] = np.array(
            [len(item_dict[iid]) if iid in item_dict else 0 for iid in range(self.item_num)])
        self.val_data[HIS_NEG_IDX] = self.generate_inter_seq_hi(
            save_dict=item_dict, group_id=self.val_data[IID], save_id=self.val_data[UID],
            filter_col=self.val_data[LABEL], col_filter=label_filter)
        self.item_data[HIS_NEG_VAL] = np.array(
            [len(item_dict[iid]) if iid in item_dict else 0 for iid in range(self.item_num)])
        self.test_data[HIS_NEG_IDX] = self.generate_inter_seq_hi(
            save_dict=item_dict, group_id=self.test_data[IID], save_id=self.test_data[UID],
            filter_col=self.test_data[LABEL], col_filter=label_filter)
        self.item_data[HIS_NEG_SEQ] = np.array(
            [np.array(item_dict[iid], dtype=int) if iid in item_dict else np.array([], dtype=int)
             for iid in range(self.item_num)], dtype=object)
        return item_dict

    def drop_neg_interactions(self, data, label_filter=lambda x: x <= 0):
        indices = label_filter(data[LABEL])
        remove = np.sum(indices)
        self.reader_logger.info('drop neg interactions: remove {}, leave {}'.format(
            remove, len(indices) - remove))
        for c in data:
            data[c] = data[c][~indices]

    def build_ui_graph(self, label_filter=lambda x: x > 0):
        uids, iids, labels = self.train_data[UID], self.train_data[IID], self.train_data[LABEL]
        uids, iids = uids[label_filter(labels)], iids[label_filter(labels)]
        uid_cnt, iid_cnt = Counter(uids), Counter(iids)
        uid_cnt = np.array(
            [uid_cnt[u] if uid_cnt[u] > 0 else 1 for u in range(self.user_num)])
        iid_cnt = np.array(
            [iid_cnt[i] if iid_cnt[i] > 0 else 1 for i in range(self.item_num)])
        values = np.array([uid_cnt[uid] * iid_cnt[iid]
                          for uid, iid in zip(uids, iids)])
        values = np.power(values, -0.5).astype(np.float32)
        rows = np.concatenate([uids, iids + self.user_num])
        cols = np.concatenate([iids + self.user_num, uids])
        values = np.concatenate([values, values])
        self.ui_graph = sp.coo_matrix((values, (rows, cols)),
                                      shape=(self.user_num + self.item_num, self.user_num + self.item_num))
        return self.ui_graph

    def prepare_cloze_train_idx(self, max_his):
        idx_map = []
        for uid in self.user_data[UID]:
            idx_map.append(uid)
            if max_his > 0:
                uid_pos_len = len(
                    self.user_data[HIS_POS_SEQ][uid][:self.user_data[HIS_POS_TRAIN][uid]])
                if uid_pos_len > max_his:
                    for i in range(round(1.0 * (uid_pos_len - max_his) / max_his)):
                        idx_map.append(uid)
        self.cloze_train_idx = np.array(idx_map)

    def count_item_popularity(self, label_filter=lambda x: x > 0):
        iids = self.train_data[IID]
        if label_filter is not None:
            iids = iids[label_filter(self.train_data[LABEL])]
        iids_cnt = Counter(iids)
        return np.array([iids_cnt[i] for i in range(self.item_num)], dtype=int)

    def count_user_warm(self, label_filter=lambda x: x > 0):
        uids = self.train_data[UID]
        if label_filter is not None:
            uids = uids[label_filter(self.train_data[LABEL])]
        uids_cnt = Counter(uids)
        return np.array([uids_cnt[u] for u in range(self.user_num)], dtype=int)
