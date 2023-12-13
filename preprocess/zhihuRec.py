# coding=utf-8



import sys
sys.path.insert(0, '../')
sys.path.insert(0, './')
import os
import re
import socket
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from fencr.utilities.io import check_mkdir
from fencr.utilities.dataset import *
from fencr.configs.settings import *
from fencr.configs.constants import *

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

RAW_DATA = '/home/zxy/DATA/zhihuRec'

USERS_FILE = os.path.join(RAW_DATA, 'user_infos.txt')
ITEMS_FILE = os.path.join(RAW_DATA, 'answer_infos.txt')
INTERACTIONS_FILE = os.path.join(RAW_DATA, 'zhihu1M.txt')

# RECSYS2017_DATA_DIR = os.path.join(DATA_DIR, 'RecSys2017')
# USER_FEATURE_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017.users.csv')
# USER_ID_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017.uid_dict.csv')
# ITEM_FEATURE_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017.items.csv')
# ITEM_ID_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017.iid_dict.csv')
# ALL_INTER_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017.all.csv')
# check_mkdir(RECSYS2017_DATA_DIR)

ZHIHU_REC = os.path.join(DATA_DIR, 'ZhihuRec')
USER_FEATURE_FILE = os.path.join(ZHIHU_REC, 'ZhihuRec.users.csv')
USER_ID_FILE = os.path.join(ZHIHU_REC, 'ZhihuRec.uid_dict.csv')
ITEM_FEATURE_FILE = os.path.join(ZHIHU_REC, 'ZhihuRec.items.csv')
ITEM_ID_FILE = os.path.join(ZHIHU_REC, 'ZhihuRec.iid_dict.csv')
ALL_INTER_FILE = os.path.join(ZHIHU_REC, 'ZhihuRec.all.csv')
check_mkdir(ZHIHU_REC)


def format_all_inter():
    interaction_dict = {'user_id': [], 'item_id': [], LABEL: [], TIME: []}
    with open(INTERACTIONS_FILE, 'r') as interaction_file:
        data = interaction_file.readline()
        while data:
            [user_id, interaction_num, interaction_infos,
                _, _] = data.split('\t')
            print(user_id, interaction_num)
            interaction_infos = interaction_infos.split(',')
            for i in range(int(interaction_num)):
                interaction_info = interaction_infos[i]
                [item_id, explosure_time,
                    read_time] = interaction_info.split('|')
                interaction_dict['user_id'].append(user_id)
                interaction_dict['item_id'].append(item_id)
                interaction_dict[TIME].append(explosure_time)
                if int(read_time) > 0:
                    interaction_dict[LABEL].append(1)
                elif int(read_time) == 0:
                    interaction_dict[LABEL].append(0)
                else:
                    raise ValueError("Invalid Read Time")
                data = interaction_file.readline()
    inter_df = pd.DataFrame(interaction_dict)
    inter_df.columns = ['user_id', 'item_id', LABEL, TIME]
    # inter_df = inter_df.dropna().astype(int)
    # inter_df[LABEL] = inter_df[LABEL].apply(lambda x: 0 if x == 4 or x == 0 else 1)
    inter_df = inter_df.sort_values(by=LABEL).reset_index(drop=True)
    inter_df = inter_df.drop_duplicates(['user_id', 'item_id'], keep='last')
    # print('origin label: ', Counter(inter_df[LABEL]))

    inter_df,uid_df,uid_dict = renumber_ids(inter_df,old_column='user_id',new_column=UID)
    uid_df.to_csv(USER_ID_FILE,sep='\t',index=False)
    inter_df,iid_df,iid_dict = renumber_ids(inter_df,old_column='item_id',new_column=IID)
    iid_df.to_csv(ITEM_ID_FILE,sep='\t',index=False)   

    # pos_df = inter_df[inter_df[LABEL] > 0]
    # pos_df, uid_df, uid_dict = renumber_ids(pos_df, old_column='user_id', new_column=UID)
    # uid_df.to_csv(USER_ID_FILE, sep='\t', index=False)
    # pos_df, iid_df, iid_dict = renumber_ids(pos_df, old_column='item_id', new_column=IID)
    # iid_df.to_csv(ITEM_ID_FILE, sep='\t', index=False)

    # neg_df = inter_df[(inter_df[LABEL] <= 0) &
    #                   (inter_df['user_id'].isin(uid_dict)) & (inter_df['item_id'].isin(iid_dict))]
    # neg_df['user_id'] = neg_df['user_id'].apply(lambda x: uid_dict[x])
    # neg_df['item_id'] = neg_df['item_id'].apply(lambda x: iid_dict[x])
    # neg_df = neg_df.rename(columns={'user_id': UID, 'item_id': IID})
    # if sample_neg == 0:
    #     neg_df = None
    # elif sample_neg < 0:
    #     neg_df = neg_df.sample(n=len(pos_df))
    # elif 0 < sample_neg < 1:
    #     neg_df = inter_df[inter_df[LABEL] == 0].sample(frac=sample_neg, replace=False)
    # elif sample_neg > 1:
    #     neg_df = inter_df[inter_df[LABEL] == 0].sample(n=int(sample_neg), replace=False)
    # inter_df = pd.concat([pos_df, neg_df]).sort_index() if neg_df is not None else pos_df

    inter_df = inter_df.sort_values(by=[TIME, UID], kind='mergesort').reset_index(drop=True)
    inter_df.to_csv(ALL_INTER_FILE, sep='\t', index=False)
    return


def format_user_feature():
    user_df = pd.read_csv(USERS_FILE, sep='\t')
    user_columns = [UID, 'u_register_time', 'u_gender', 'u_visit_freq', 'u_follow_user_num',
                    'u_follow_topic_num', 'u_follow_question_num', 'user_answer_num','user_question_num','u_review_num',
                    'u_answer_like_num','u_answer_review_num','u_answer_agree_num', 'u_answer_oppose_num',
                    'u_register_type','u_register_platform','u_android','u_iphone','u_ipad',
                    'u_pc','u_web','u_device_model','u_device_brand','u_platform',
                    'u_province','u_city','u_follow_topic']
    user_df.columns = user_columns
    user_df = user_df.drop_duplicates(UID)
    uid_dict = read_id_dict(
        USER_ID_FILE, key_column='user_id', value_column=UID)
    user_df = user_df[user_df[UID].isin(uid_dict)]
    user_df[UID] = user_df[UID].apply(lambda x: uid_dict[x])

    # drop the columns of register time and follow_topics
    u_drop_columns = ['u_register_time', 'u_follow_topic']
    user_df = user_df.drop(columns=u_drop_columns)

    # # u_wtcj_c, u_premium_c
    # user_df['u_wtcj_c'] = user_df['u_wtcj_c'] + 1
    # user_df['u_premium_c'] = user_df['u_premium_c'] + 1

    # u_country_c
    # country_dict = {'non_dach': 0, 'de': 1, 'at': 2, 'ch': 3}
    # user_df['u_country_c'] = user_df['u_country_c'].apply(
    #     lambda x: country_dict[x])

    user_df = user_df.sort_values(UID).reset_index(drop=True)
    user_df.index = user_df[UID]
    user_df.loc[0] = 0
    user_df = user_df.sort_index()
    # print(user_df)
    # user_df.info(null_counts=True)
    user_df.to_csv(USER_FEATURE_FILE, sep='\t', index=False)
    return user_df


def format_item_feature():
    item_df = pd.read_csv(ITEMS_FILE, sep='\t')
    item_columns = [IID, 'i_question_id', 'i_anonymous', '', 'i_industry_id_c',
                    'i_country_c', 'i_is_paid_c', 'i_region_c', 'i_latitude_c', 'i_longitude_c',
                    'i_employment_c', 'i_tags_s', 'i_created_at_c']
    item_df.columns = item_columns
    item_df = item_df.drop_duplicates(IID)
    iid_dict = read_id_dict(
        ITEM_ID_FILE, key_column='item_id', value_column=IID)
    item_df = item_df[item_df[IID].isin(iid_dict)]
    item_df[IID] = item_df[IID].apply(lambda x: iid_dict[x])

    # i_title_s, i_tags_s
    i_drop_columns = ['i_title_s', 'i_tags_s']
    item_df = item_df.drop(columns=i_drop_columns)

    # i_is_paid_c
    item_df['i_is_paid_c'] = item_df['i_is_paid_c'] + 1

    # i_country_c
    country_dict = {'non_dach': 0, 'de': 1, 'at': 2, 'ch': 3}
    item_df['i_country_c'] = item_df['i_country_c'].apply(
        lambda x: country_dict[x])

    # i_latitude_c, i_longitude_c
    item_df['i_latitude_c'] = item_df['i_latitude_c'].apply(
        lambda x: 0 if np.isnan(x) else int((int(x + 90) / 10)) + 1)
    item_df['i_longitude_c'] = item_df['i_longitude_c'].apply(
        lambda x: 0 if np.isnan(x) else int((int(x + 180) / 10)) + 1)

    # i_created_at_c
    item_df['i_created_at_c'] = pd.to_datetime(
        item_df['i_created_at_c'], unit='s')
    item_year = item_df['i_created_at_c'].apply(lambda x: x.year)
    min_year = item_year.min()
    item_month = item_df['i_created_at_c'].apply(lambda x: x.month)
    item_df['i_created_at_c'] = (
        item_year.fillna(-1) - min_year) * 12 + item_month.fillna(-1)
    item_df['i_created_at_c'] = item_df['i_created_at_c'].apply(
        lambda x: int(x) if x > 0 else 0)

    item_df = item_df.sort_values(IID).reset_index(drop=True)
    item_df.index = item_df[IID]
    item_df.loc[0] = 0
    item_df = item_df.sort_index()
    # print(item_df)
    # item_df.info(null_counts=True)
    item_df.to_csv(ITEM_FEATURE_FILE, sep='\t', index=False)
    return


def main():
    format_all_inter()
    format_user_feature()
    format_item_feature()

    # dataset_name = 'recsys2017-1-1'
    # leave_out_csv(ALL_INTER_FILE, dataset_name, warm_n=1, leave_n=1, max_user=10000)

    dataset_name = 'recsys2017-5-1'
    leave_out_csv(ALL_INTER_FILE, dataset_name,
                  warm_n=5, leave_n=1, max_user=10000)

    copy_ui_features(dataset_name=dataset_name,
                     user_file=USER_FEATURE_FILE, item_file=ITEM_FEATURE_FILE)
    random_sample_eval_iids(dataset_name, sample_n=1000, include_neg=False)
    # random_sample_eval_iids(dataset_name, sample_n=1000, include_neg=True)
    return


if __name__ == '__main__':
    # main()
    format_all_inter()
