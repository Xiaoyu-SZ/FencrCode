
# coding=utf-8
import matplotlib.pyplot as plt
import os
import logging
import pickle
import socket
import datetime
import time
from argparse import ArgumentParser
from numpy import mask_indices
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
import numpy as np
import os

from fencr.models.FENCR import FEATURE_VALUES

from ..utilities.logging import format_log_metrics_dict
from ..models import *
from ..tasks.Task import Task
from ..configs.settings import *
from ..configs.constants import *
from ..utilities.logging import DEFAULT_LOGGER, metrics_list_best_iter
from ..utilities.io import check_mkdir

MOVIELEN_FEATURES = ["USER_ID{0}".format(i) for i in range(8)]+["ITEM_ID{0}".format(i) for i in range(8)]+["u_age_c", "u_gender_c", "u_occupation_c", "i_year_c", "i_Action_c", "i_Adventure_c", "i_Animation_c",	"i_Children's_c", "i_Comedy_c", "i_Crime_c",
                                                                                                        "i_Documentary_c", "i_Drama_c", "i_Fantasy_c", "i_Film-Noir_c", "i_Horror_c",	"i_Musical_c",	"i_Mystery_c", "i_Romance_c", "i_Sci-Fi_c", "i_Thriller_c", "i_War_c", "i_Western_c", "i_Other_c"]

RECSYS_FEATURES = ["USER_ID{0}".format(i) for i in range(1)]+["ITEM_ID{0}".format(i) for i in range(1)] + \
["u_career_level_c", "u_discipline_id_c", "u_industry_id_c", "u_country_c", "u_region_c", "u_experience_n_entries_class_c",
    "u_experience_years_experience_c", "u_experience_years_in_current_c", "u_edu_degree_c", "u_wtcj_c", "u_premium_c"] + \
['i_career_level_c', 'i_discipline_id_c', 'i_industry_id_c', 'i_country_c', 'i_is_paid_c',
    'i_region_c', 'i_latitude_c', 'i_longitude_c', 'i_employment_c', 'i_created_at_c']

# https://tianchi.aliyun.com/dataset/dataDetail?dataId=56
TAOBAO_FEATURES = ['USER_ID','ITEM_ID']+\
['u_cms_segid','u_group_id','u_gender','u_age','u_p_level','u_shop_level','u_occupation','user_class_level'] + \
['i_cate_id','i_campaign_id,','i_customer_id','i_brand','i_price']
# 用户id, 广告id （2项）
# 用户Feature：8项
# 微群id（好多都是0）， groupid， 性别1-2，年龄1-6（？），消费层次1-3（也有很多0），购物深度1-3，是否为大学生（1-2）， 城市层级
# 物品Feature: 5项
# 商品类型id，广告计划id，广告主id，品牌id，价格
# TODO：价格这项怎么处理（画分布图？）





class CaseStudy(Task):
    @ staticmethod
    def add_task_args(parent_parser):
        parser = Task.add_task_args(parent_parser)
        parser.add_argument('--ckpt_v', type=str, default='',
                            help='If not none, load model from ckpt version, ignore training')
        parser.add_argument('--save_f', type=str, default='',
                            help='Save predict results as pickle file to specific path, default to model checkpoint path')
        return parser

    def __init__(self, ckpt_v, save_f, *args, **kwargs):
        self.ckpt_v = ckpt_v
        self.save_f = save_f
        super().__init__(*args, **kwargs)

    def parse_model_to_rule(self, model):
        model.model_logger.info("Rule sets formed by FENCR:")
        N_feature = model.select_layers[0]().shape[1]
        edge_num = list()
        rule_set = list()
        model.model_logger.info("#feature num(including id):", str(N_feature))
        # weight = model.weight_layer_layer.weight.clone().detach()[0]
        if 'recsys' in self.dataset:
            print('it is a recsys dataset')
            features = RECSYS_FEATURES
        elif 'taobao' in self.dataset:
            print('it is a taobao dataset')
            features = TAOBAO_FEATURES  
            
        list_count = np.zeros(len(features))
        poses = []

        for li, size in enumerate(model.layers):
            for i in range(size*2):
                # if i < size:
                #     print(
                #         "the AND NODE {0} in Layer {1} is connected with:".format(i, li))
                # else:
                #     print(
                #         "the OR NODE {0} in Layer {1} is connected with:".format(i, li))
                mask = model.select_layers[li]()[i]
                pos = torch.where(mask == 1)[0].tolist()
                poses.append(pos)
                
                vars = [features[k] for k in pos]
                list_count[pos] = list_count[pos]+1
      
                # if 'recsys' in self.dataset:
                #     print('it is a recsys dataset')
                #     vars = [RECSYS_FEATURES[k] for k in pos]
                # elif 'taobao' in self.dataset:
                #     print('it is a taobao dataset')
                #     vars = [TAOBAO_FEATURES[k] for k in pos]
                # elif 'movielen' in self.dataset:
                #     raise ValueError("movielen datasets are not supported yet")
                # else:
                #     raise ValueError("dataset {0} is not supported yet".format(self.dataset))
                if i < size:
                    rule_set.append(' |and|  '.join(vars))
                    model.model_logger.info("#Rule{0}: ".format(i)+' |and|  '.join(vars))
                else:
                    rule_set.append(' |or|  '.join(vars))
                    model.model_logger.info("#Rule{0}: ".format(i)+' |or|  '.join(vars))
                # model.model_logger.info("#Weight: "+str(weight[i].item()))
                edge_num.append(len(pos))
        
        fontsize=20
                
        model.model_logger.info(edge_num)
        model.model_logger.info("average edge num:"+str(np.mean(np.array(edge_num))))
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(features)), list_count)
        ax.set_ylim([0,32])
        ax.set_ylabel('times',fontdict={'fontsize':fontsize})
        ax.set_xlabel('feature index',fontdict={'fontsize':fontsize})
        # ax.set_ylabel('fruit supply')
        # ax.set_title('Fruit supply by kind and color')
        # ax.legend(title='Fruit color')
        import json
        with open(os.path.join(model.log_dir,'poses.json')) as f:
            json.dump(poses,f)
        plt.savefig(os.path.join(model.log_dir,'rule_feature_distribution.png'))
        exit(0)
        

        

    def run(self, *args, **kwargs):
        start_time = time.time()
        self._init_environment()
        model = self._init_model()

        # # init logger
        version = os.path.join(
            self.model_name, self.ckpt_v) if self.ckpt_v != '' else None
        save_dir, name, version = model.init_logger(version=version)
        self.task_logger = model.model_logger

        # # init trainer
        trainer_args = DEFAULT_TRAINER_ARGS
        if self.trainer_args is not None:
            trainer_args = {**trainer_args, **self.trainer_args}
        model.init_trainer(save_dir=save_dir, name=name,
                            version=version, **trainer_args)

        if self.ckpt_v == '':
            # # train
            model.model_logger.info("ckpt is needed to be provided")
            raise ValueError('Empty CKPT.')
            model.fit()

        else:
            # # load model
            checkpoint_path = os.path.join(
                model.log_dir, CKPT_DIR, CKPT_F + '.ckpt')
            hparams_file = os.path.join(model.log_dir, 'hparams.yaml')
            model.load_model(checkpoint_path=checkpoint_path,
                                hparams_file=hparams_file)
        self.parse_model_to_rule(model)

        # # test
        predict_result = model.predict(model.get_dataset(phase=PREDICT_PHASE))
        
        weights = []
        for predict in predict_result:
            weight = predict['weight'].view(-1,32)
            weights.append(weight)
        weights = torch.cat(weights,0)
        for i in range(32):
            model.TensorboardWriter.add_histogram(
                'weight distribution'+str(i), weights[:,i], 0)
        model.TensorboardWriter.close()
        weights = weights.cpu().numpy()
        
        np.save(os.path.join(model.log_dir,'weight.npy'),weights)

        
        end_time = time.time()

        # # format result
        best_iter = metrics_list_best_iter(model.val_metrics_buf)
        interval = trainer_args['val_check_interval']
        task_result = {
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'time': str(datetime.timedelta(seconds=int(end_time - start_time))),
            'server': socket.gethostname(),
            'model_name': self.model_name,
            'version': version.split('/')[-1],
            'num_para': model.count_variables(),
            'best_iter': best_iter * interval,
            'train_metrics': format_log_metrics_dict(
                model.train_metrics_buf[int(best_iter * interval - 1)]) if len(model.train_metrics_buf) > 0 else '',
            'val_metrics': format_log_metrics_dict(
                model.val_metrics_buf[best_iter]) if len(model.val_metrics_buf) > best_iter else '',
            'test_metrics': format_log_metrics_dict(
                model.test_metrics_buf[-1] if len(model.test_metrics_buf) > 0 else ''),
        }
        self.task_logger.info('')
        for key in task_result:
            self.task_logger.info(key + ': ' + str(task_result[key]))
        save_f = self.save_f if self.save_f != '' else os.path.join(
            model.log_dir, PREDICT_F)
        check_mkdir(save_f)
        torch.save(predict_result, open(save_f, 'wb'), pickle_protocol=pickle.HIGHEST_PROTOCOL,
                    _use_new_zipfile_serialization=False)
        return predict_result

        #         if i < size:
        #             print(
        #                 "the AND NODE {0} in Layer {1} is connected with:".format(i, li))
        #         else:
        #             print(
        #                 "the OR NODE {0} in Layer {1} is connected with:".format(i, li))
        #
        #         print(mask[i])
