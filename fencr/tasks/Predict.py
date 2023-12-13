# coding=utf-8

import os
import logging
import pickle
import socket
import datetime
import time
from argparse import ArgumentParser
from numpy import mask_indices
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import numpy as np

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


def parse_model(model):
    print("Do Explanation Analysis")
    N_feature = model.select_layers[0]().shape[1]
    edge_num = list()
    print("N_feature:", N_feature)
    for li, size in enumerate(model.layers):
        for i in range(size*2):
            if i < size:
                print(
                    "the AND NODE {0} in Layer {1} is connected with:".format(i, li))
            else:
                print(
                    "the OR NODE {0} in Layer {1} is connected with:".format(i, li))
            mask = model.select_layers[li]()[i]
            print(mask)
            position = torch.where(mask == 1)
            print(list(position))
            edge_num.append(list(position)[0].shape[0])
    print(edge_num)
    print("average edge num:", np.mean(np.array(edge_num)))
# def parse_model(model):
#     logic_text = list()
#     for layer in model.layers:
#         logic_text.append([""]*(layer*2))
#     print(logic_text)

#     print(model.out_layer.weight)

#     for li, size in enumerate(model.layers):
#         for i in range(size*2):
#             mask = model.select_layers[li]()
#             print(mask.shape)

#             pos = torch.where(mask[i] == 1)[0]
#             print(pos)
#             if li == 0:
#                 vars = ["({0})".format(RECSYS_FEATURES[pos[k]])
#                         for k in range(pos.shape[0])]
#             else:
#                 vars = ["({0})".format(logic_text[li-1][pos[k]])
#                         for k in range(pos.shape[0])]
#             if i < size:
#                 logic_text[li][i] = " and ".join(vars)
#             else:
#                 logic_text[li][i] = " or ".join(vars)

#             print(logic_text[li][i])

# def load_local(save_f):
#     predict_resullt = torch.load(predict_result, open(save_f, 'wb'), pickle_protocol=pickle.HIGHEST_PROTOCOL,
#                    _use_new_zipfile_serialization=False)


class Predict(Task):
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
            model.fit()
        else:
            # # load model
            checkpoint_path = os.path.join(
                model.log_dir, CKPT_DIR, CKPT_F + '.ckpt')
            hparams_file = os.path.join(model.log_dir, 'hparams.yaml')
            model.load_model(checkpoint_path=checkpoint_path,
                             hparams_file=hparams_file)
        parse_model(model)

        # # test
        predict_result = model.predict(model.get_dataset(phase=PREDICT_PHASE))
        # print(predict_result[0])
        for i in range(len(predict_result)):
            feature_values = predict_result[i]['feature_values']
            last_layer_output = predict_result[i]['last_layer_output']
            prediction = predict_result[i]['prediction']
            # print(feature_values.shape)  # batch * S * feature
            # print(last_layer_output.shape)  # batch * S * out_layer

            B = feature_values.shape[0]
            S = feature_values.shape[1]

            # for j in range(B):
            #     for k in range(S):
            #         score = prediction[i]
            #         print(score)
            #         for li, size in enumerate(model.layers):
            #             mask = model.select_layers[li].get_select_01()
            #             for ii in range(size*2):
            #                 print(mask[ii])
            #                 connect_pos = torch.where(mask[ii]!=0)[0]
            #                 print(connect_pos)
            #                 print("rule{0}:".format(ii))
            #                 if i < size:
            #                     print("and".join(connect_pos))
            #                 else:
            #                     print("or".join(connect_pos))
            #                 print(last_layer_output[j][k][ii])

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
