# # coding=utf-8


import sys
import os
import re
import socket
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

sys.path.insert(0, '../')
sys.path.insert(0, './')
from fencr.configs.constants import *
from fencr.configs.settings import *
from fencr.utilities.dataset import *
from fencr.utilities.io import check_mkdir
np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

RAW_DATA = '~/DATA/taobao/'

USERS_FILE = os.path.join(RAW_DATA, 'user_profile.csv')
ITEMS_FILE = os.path.join(RAW_DATA, 'ad_feature.csv')
INTERACTIONS_FILE = os.path.join(RAW_DATA, 'raw_sample.csv')

TAOBAO_DATA_DIR = os.path.join(DATA_DIR, 'Taobao')
USER_FEATURE_FILE = os.path.join(TAOBAO_DATA_DIR, 'Taobao.users.csv')
USER_ID_FILE = os.path.join(TAOBAO_DATA_DIR, 'Taobao.uid_dict.csv')
ITEM_FEATURE_FILE = os.path.join(TAOBAO_DATA_DIR, 'Taobao.items.csv')
ITEM_ID_FILE = os.path.join(TAOBAO_DATA_DIR, 'Taobao.iid_dict.csv')
ALL_INTER_FILE = os.path.join(TAOBAO_DATA_DIR, 'Taobao.all.csv')
check_mkdir(TAOBAO_DATA_DIR)


def format_all_inter():
    inter_df = pd.read_csv(INTERACTIONS_FILE, sep=',')
    inter_df.columns = ['user_id', TIME, 'item_id', 'pid', 'noclk', LABEL]
    inter_drop_columns = ['pid', 'noclk']
    inter_df = inter_df.drop(columns=inter_drop_columns)
    inter_df = inter_df.dropna().astype(int)
    # inter_df[LABEL] = inter_df[LABEL].apply(lambda x: 0 if x == 4 or x == 0 else 1)
    inter_df = inter_df.sort_values(by=LABEL).reset_index(drop=True)
    inter_df = inter_df.drop_duplicates(['user_id', 'item_id'], keep='last')
    # print('origin label: ', Counter(inter_df[LABEL]))

    iid_dict = read_id_dict(
        ITEM_ID_FILE, key_column='item_id', value_column=IID)
    inter_df = inter_df[inter_df['item_id'].isin(iid_dict)]
    inter_df['item_id'] = inter_df['item_id'].apply(lambda x: iid_dict[x])

    uid_dict = read_id_dict(
        USER_ID_FILE, key_column='user_id', value_column=UID)
    inter_df = inter_df[inter_df['user_id'].isin(uid_dict)]
    inter_df['user_id'] = inter_df['user_id'].apply(lambda x: uid_dict[x])
    inter_df = inter_df.rename(columns={'user_id': UID, 'item_id': IID})


    inter_df = inter_df.sort_values(
        by=[TIME, UID], kind='mergesort').reset_index(drop=True)
    inter_df.to_csv(ALL_INTER_FILE, sep='\t', index=False)
    return


def format_user_feature():
    user_df = pd.read_csv(USERS_FILE, sep=',')
    user_columns = ['user_id', 'u_cms_segid_c', 'u_cms_group_id_c', 'u_final_gender_code_c', 'u_age_level_c',
                    'u_pvalue_level_c', 'u_shopping_level_c', 'u_occupation_c', 'u_new_user_class_level_c']
    user_df.columns = user_columns # shape 1061768 * 9
    user_df = user_df.drop_duplicates('user_id')
    user_df, uid_df, uid_dict = renumber_ids(user_df, old_column='user_id', new_column=UID)
    uid_df.to_csv(USER_ID_FILE, sep='\t', index=False)
    user_df['u_occupation_c'] = user_df['u_occupation_c'] + 1

    user_df = user_df.fillna(0)
    user_df = user_df.astype(int)

    user_df = user_df.sort_values(UID).reset_index(drop=True)
    user_df.index = user_df[UID]
    user_df.loc[0] = 0
    user_df = user_df.sort_index()
    user_df.info(null_counts=True)
    user_df.to_csv(USER_FEATURE_FILE, sep='\t', index=False)
    return user_df

def format_item_feature():
    item_df = pd.read_csv(ITEMS_FILE, sep=',')
    item_columns = ['item_id', 'i_cate_id_c', 'i_campaign_id_c',
                    'i_customer_id_c', 'i_brand_c', 'i_price_c']
    item_df.fillna(0)
    item_df.columns = item_columns
    item_df = item_df.drop_duplicates('item_id')

    item_df, iid_df, iid_dict = renumber_ids(
        item_df, old_column='item_id', new_column=IID)
    iid_df.to_csv(ITEM_ID_FILE, sep='\t', index=False)

    # iid_dict = read_id_dict(
    #     ITEM_ID_FILE, key_column='item_id', value_column=IID)
    # item_df = item_df[item_df[IID].isin(iid_dict)]
    # item_df[IID] = item_df[IID].apply(lambda x: iid_dict[x])

    # # i_title_s, i_tags_s
    # i_drop_columns = ['i_title_s', 'i_tags_s']
    # item_df = item_df.drop(columns=i_drop_columns)

    # # i_is_paid_c
    # item_df['i_is_paid_c'] = item_df['i_is_paid_c'] + 1

    # # i_country_c
    # country_dict = {'non_dach': 0, 'de': 1, 'at': 2, 'ch': 3}
    # item_df['i_country_c'] = item_df['i_country_c'].apply(lambda x: country_dict[x])

    # i_latitude_c, i_longitude_c
    # visualize_price_distribution(item_df['i_price_c'])
    # exit(0)
    item_df['i_price_c'] = item_df['i_price_c'].apply(
        lambda x: 0 if np.isnan(x) else int((int(x) / 1000)) + 1)

    item_df = item_df.fillna(0)
    item_df = item_df.astype(int)
    item_df = item_df.sort_values(IID).reset_index(drop=True)
    item_df.index = item_df[IID]
    item_df.loc[0] = 0
    item_df = item_df.sort_index()
    item_df.to_csv(ITEM_FEATURE_FILE, sep='\t', index=False)
    return


def main():
    # format_all_inter()
    format_user_feature()
    format_item_feature()
    format_all_inter()

    # dataset_name = 'recsys2017-1-1'
    # leave_out_csv(ALL_INTER_FILE, dataset_name, warm_n=1, leave_n=1, max_user=10000)

    dataset_name = 'taobao-1-1'
    leave_out_csv(ALL_INTER_FILE, dataset_name,
                  warm_n=1, leave_n=1, max_user=10000)

    copy_ui_features(dataset_name=dataset_name,
                     user_file=USER_FEATURE_FILE, item_file=ITEM_FEATURE_FILE)
    random_sample_eval_iids(dataset_name, sample_n=1000, include_neg=False)
    # random_sample_eval_iids(dataset_name, sample_n=1000, include_neg=True)
    return


if __name__ == '__main__':
    main()


