# coding=utf-8
import torch.nn

from ..configs.constants import *
from .RecModel import RecModel
from ..modules.nas import GumbelSoftmax
from .Model import Model
import numpy as np
import pdb

USER_MH = 'user_mh'
USER_NU = 'user_nm'
ITEM_MH = 'item_mh'
ITEM_NU = 'item_nm'
CTXT_MH = 'ctxt_mh'
CTXT_NU = 'ctxt_nm'

REGULARIZER_VEC = 'regularizer_vec'
REGULARIZER_SELECT = 'regularizer_select'
LOGIC_REGULARIZER = 'logic_regularizer'
LAST_LAYER_OUTPUT = 'last_layer_output'
FEATURE_VALUES = 'feature_values'
RECSYS_USER_FEATURE_NUM = 12
RECSYS_ITEM_FEATURE_NUM = 11
TAOBAO_USER_FEATURE_NUM = 8
TAOBAO_ITEM_FEATURE_NUM = 5
LOSS_NUM = 9


class SelectLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, gumbel_r, gumbel_tau, gumbel_hard):
        super().__init__()
        self.gumbel_softmax = GumbelSoftmax(
            random_r=gumbel_r, tau=gumbel_tau, hard_r=gumbel_hard, hard=True)
        self.select = torch.nn.Parameter(torch.empty(output_size, input_size).uniform_(0, 1),
                                         requires_grad=True)  # out * in

    def forward(self):
        select_point = torch.stack(
            (self.select, -self.select + 1), dim=-1)  # out * in * 2
        select_point = self.gumbel_softmax(select_point)  # out * in * 2
        select_point = select_point[..., 0]  # out * in
        select_all = self.gumbel_softmax(self.select)  # out * in
        select = -(-select_point + 1) * (-select_all + 1) + 1
        return select

    def get_select_01(self):
        select_point = torch.stack(
            (self.select, -self.select + 1), dim=-1)  # out * in * 2
        select_point = self.gumbel_softmax(select_point)  # out * in * 2
        select_point = select_point[..., 0]  # out * in
        return select_point


class AndOrLayer(torch.nn.Module):
    def __init__(self, vec_size):
        super().__init__()
        self.vec_size = vec_size
        self.value_trans = torch.nn.Sequential(
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, self.vec_size, bias=False),
        )
        self.final_trans = torch.nn.Sequential(
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, self.vec_size, bias=False),
        )

    def forward(self, vectors, select=None):  # B * S * L * f * v, B * S * L * f
        if select is None:
            L = vectors.shape[-3]
            f = vectors.shape[-2]
            select = torch.ones(L, f).to(vectors.device)
        vectors = self.value_trans(vectors)  # B * S * L * f * v
        cross = vectors.sum(
            dim=-2).pow(2) - vectors.pow(2).sum(dim=-2) + vectors.sum(dim=-2)  # ? * v
        cross_n = select.sum(dim=-1, keepdim=True)  # B * S * L * 1(1的个数？)
        cross = cross / cross_n.pow(2)  # ? * v
        results = self.final_trans(cross)  # ? * v
        return results  # B * S * L * v


class FENCR(RecModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        """
        parser = RecModel.add_model_specific_args(parent_parser)
        parser.add_argument('--gumbel_r', type=float, default=0.5,
                            help='Random explore ratio random_r in Gumbel softmax (default: 0.5)')
        parser.add_argument('--gumbel_tau', type=float, default=1.0,
                            help='Temperature tau in Gumbel softmax (default: 1.0)')
        parser.add_argument('--gumbel_hard', type=float, default=1.0,
                            help='Whether use hard Gumbel softmax (default: 1)')
        parser.add_argument('--logic_tau', type=float, default=8.0,
                            help='The larger, the logic is harder (default: 1.0)')
        parser.add_argument('--layers', type=str, default='[8]',
                            help='Hidden layers in the deep part.')
        parser.add_argument('--latent_dim', type=int, default=1,
                            help='Latent number of dims of ID features.')
        parser.add_argument('--att_size', type=int, default=16,
                            help='Size of attention layer.')
        parser.add_argument('--sim_scale', type=int, default=10,
                            help='Expand the raw similarity *sim_scale before sigmoid.')
        parser.add_argument('--r_logic', type=float, default=0.0,
                            help='Weight of logic regularizer loss')
        parser.add_argument('--r_length', type=float, default=0.0,
                            help='Weight of vector length regularizer loss')
        parser.add_argument('--r_mask', type=float, default=0.0,
                            help='Weight of mask loss')
        parser.add_argument('--use_id', type=int, default=1,
                            help='Whether use id information (default: 1)')
        parser.add_argument('--bucket_size', type=int, default=20,
                            help='Bucket number of id representation.')
        parser.add_argument('--adaptive_loss', type=int, default=0,
                            help='Whether use adaptive weight when counting loss of .')
        parser.add_argument('--output_strategy', type=str, default='adaptive_sigmoid_ui',
                            help='Strategy used in output layer, chosen from [sum, voting, adaptive]')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Hidden size in weight layer for output.')
        parser.add_argument('--enable_f',type=str,default='ui',
                            help = 'Use features in which side(user/item)')
        return parser

    def __init__(self, multihot_f_num: int = None, multihot_f_dim: int = None, latent_dim: int = 8, att_size: int = 16,
                 layers: str = '[8]', gumbel_r: float = 0.1, gumbel_tau: float = 1.0,
                 gumbel_hard: float = 1.0, logic_tau: float = 8.0,
                 sim_scale=10, r_logic=0.0, r_length=0.0, r_mask=0.0, use_id=1, bucket_size: int = 20,
                 adaptive_loss: int = 0, output_strategy: str = 'sum', hidden_size: int = 64, 
                 enable_f: str = 'ui',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihot_f_num = multihot_f_num
        self.multihot_f_dim = multihot_f_dim
        self.latent_dim = latent_dim
        self.att_size = att_size
        self.layers = eval(layers) if type(layers) is str else layers
        self.gumbel_r = gumbel_r
        self.gumbel_tau = gumbel_tau
        self.gumbel_hard = gumbel_hard
        self.logic_tau = logic_tau
        self.sim_scale = sim_scale
        self.r_logic = r_logic
        self.r_length = r_length
        self.r_mask = r_mask
        self.use_id = use_id
        self.bucket_size = bucket_size
        self.adaptive_loss = adaptive_loss
        self.output_strategy = output_strategy
        self.hidden_size = hidden_size
        self.gumbel_softmax = GumbelSoftmax(
            random_r=gumbel_r, tau=gumbel_tau, hard_r=gumbel_hard, hard=True)
        self.enable_f = enable_f

    def read_data(self, *args, **kwargs):
        reader = super().read_data(*args, **kwargs)
        reader.prepare_user_features(
            include_uid=False, multihot_features=USER_MH, numeric_features=USER_NU)
        reader.prepare_item_features(
            include_iid=False, multihot_features=ITEM_MH, numeric_features=ITEM_NU)
        reader.prepare_ctxt_features(
            include_time=False, multihot_features=CTXT_MH, numeric_features=CTXT_NU)
        self.multihot_f_num = reader.user_multihot_f_num + \
            reader.item_multihot_f_num + reader.ctxt_multihot_f_num
        self.multihot_f_dim = reader.user_multihot_f_dim + \
            reader.item_multihot_f_dim + reader.ctxt_multihot_f_dim
        return reader

    def read_formatters(self, formatters=None) -> dict:
        current = {
            '^({}|{}|{}).*({})$'.format(USER_F, ITEM_F, CTXT_F, CAT_F): None
        }
        if formatters is None:
            return super().read_formatters(current)
        return super().read_formatters({**current, **formatters})

    def dataset_get_item(self, dataset, index: int) -> dict:
        if dataset.buffer_ds > 0:
            return dataset.index_buffer[index]
        index_dict = super().dataset_get_item(dataset=dataset, index=index)
        reader = dataset.reader
        if USER_MH in reader.user_data:
            index_dict[USER_MH] = reader.user_data[USER_MH][index_dict[UID]]
        if ITEM_MH in reader.item_data:
            index_dict[ITEM_MH] = reader.item_data[ITEM_MH][index_dict[IID]
                                                            ] + reader.user_multihot_f_dim
        if CTXT_MH in dataset.data:
            index_dict[CTXT_MH] = dataset.data[CTXT_MH][index] + \
                reader.user_multihot_f_dim + reader.item_multihot_f_dim
        return index_dict

    def init_modules(self, *args, **kwargs) -> None:
        if self.bucket_size > 0:
            self.uid_id2bucket_embeddings = torch.nn.Embedding(
                self.user_num, self.bucket_size)
            self.iid_id2bucket_embeddings = torch.nn.Embedding(
                self.item_num, self.bucket_size)
            self.uid_bucket2vec_embeddings = torch.nn.Embedding(
                self.bucket_size, self.vec_size)
            self.iid_bucket2vec_embeddings = torch.nn.Embedding(
                self.bucket_size, self.vec_size)
        else:
            self.uid_embeddings = torch.nn.Embedding(
                self.user_num, self.vec_size * self.latent_dim)
            self.iid_embeddings = torch.nn.Embedding(
                self.item_num, self.vec_size * self.latent_dim)

        self.feature_embeddings = torch.nn.Embedding(
            self.multihot_f_dim, self.vec_size)

        self.select_layers = torch.nn.ModuleList()
        #  TODO: use for adding ablation, poor implementation
        if self.enable_f == 'ui':
            feature_num = self.multihot_f_num
        elif self.enable_f == 'u':
            if self.multihot_f_num == 21:
                feature_num = 11
            elif self.multihot_f_num == 13:
                feature_num = 8
            else:
                print(self.multihot_f_num)
                raise ValueError("feature num not supported.")
        elif self.enable_f == 'i':
            if self.multihot_f_num == 21:
                feature_num = 10
            elif self.multihot_f_num == 13:
                feature_num = 5
            else:
                raise ValueError("feature num not supported.")
        else:
            raise ValueError('Wrong enable_f value.')
                  
        if self.use_id:
            if self.bucket_size > 0:
                pre_size = 2 * self.bucket_size + feature_num
            else:
                pre_size = 2 * self.latent_dim + feature_num 
        else:
            pre_size = feature_num
        for size in self.layers:
            self.select_layers.append(
                SelectLayer(input_size=pre_size, output_size=size * 2,
                            gumbel_r=self.gumbel_r, gumbel_tau=self.gumbel_tau, gumbel_hard=self.gumbel_hard)
            )
            pre_size = size * 2

        self.and_layer = AndOrLayer(vec_size=self.vec_size)
        self.or_layer = AndOrLayer(vec_size=self.vec_size)
        self.neg_layer = torch.nn.Sequential(
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, self.vec_size, bias=False),
        )
        if self.output_strategy == 'voting':
            self.weight = torch.nn.Parameter(torch.empty(
                1, pre_size).uniform_(0, 1), requires_grad=True)
        elif 'adaptive_sigmoid' in self.output_strategy:
            if 'ui' in self.output_strategy:
                embed_size = 2*self.vec_size
            else:
                embed_size = self.vec_size

            self.weight_layer = torch.nn.Sequential(
                torch.nn.Linear(embed_size, self.hidden_size),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hidden_size, self.layers[-1]*2),
                torch.nn.Sigmoid()
            )
        elif 'adaptive_softmax' in self.output_strategy:
            if 'ui' in self.output_strategy:
                embed_size = 2*self.vec_size
            else:
                embed_size = self.vec_size

            self.weight_layer = torch.nn.Sequential(
                torch.nn.Linear(2*self.vec_size, self.hidden_size),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hidden_size, self.layers[-1]*2),
                torch.nn.Softmax(dim=-1)
            )
        self.true = torch.nn.Parameter(
            torch.empty(1, self.vec_size).uniform_(0, 1), requires_grad=False)
        self.layer_norm = torch.nn.LayerNorm(self.vec_size)
        self.init_weights()
        self.loss_weights = torch.nn.Parameter(
            torch.ones(LOSS_NUM), requires_grad=True)  # out * in
        self.train_step = 0
        return

    def concat_features(self, batch):
        i_ids = batch[IID]  # B * S
        sample_n = i_ids.size(-1)  # =S
        mh_features = []
        if USER_MH in batch:
            if 'u' in self.enable_f:
                mh_features.append(
                    batch[USER_MH].expand(-1, sample_n, -1))  # B * S * uf
        if ITEM_MH in batch:
            if 'i' in self.enable_f:
                mh_features.append(batch[ITEM_MH])  # B * S * if
        if CTXT_MH in batch:
            mh_features.append(batch[CTXT_MH].unsqueeze(
                dim=1).expand(-1, sample_n, -1))  # B * S * cf
        mh_features = torch.cat(
            mh_features, dim=-1) if len(mh_features) > 0 else None  # B * S * mh
        return mh_features

    def similarity(self, vector1, vector2, sigmoid=True):
        result = torch.nn.functional.cosine_similarity(
            vector1, vector2, dim=-1)
        result = result * self.sim_scale
        return result.sigmoid() if sigmoid else result

    def uniform_size(self, vector1, vector2):
        if vector1.nelement() < vector2.nelement():
            vector1 = vector1.expand_as(vector2)
        else:
            vector2 = vector2.expand_as(vector1)
        return vector1, vector2

    def form_id_embedding(self, u_ids, i_ids):
        # u_ids, B * 1 ; i_ids, B * S

        uid_buckets = self.uid_id2bucket_embeddings(
            u_ids)  # B * 1 * self.bucket_size

        uid_buckets = self.gumbel_form_bin(
            uid_buckets)  # B * 1 * self.bucket_size
        # # embedding self.bue
        uid_buckets_bin = uid_buckets.unsqueeze(
            dim=-1)  # B * 1 * self.bucket_size * 1
        uid_vecs = self.uid_bucket2vec_embeddings.weight * \
            uid_buckets_bin  # weight's shape self.bucket_size * self.vec

        iid_buckets = self.iid_id2bucket_embeddings(
            i_ids)  # B * S * self.bucket_size
        iid_buckets = self.gumbel_form_bin(
            iid_buckets)  # B * 1 * self.bucket_size
        iid_buckets_bin = iid_buckets.unsqueeze(
            dim=-1)  # B * 1 * self.bucket_size * 1
        iid_vecs = self.iid_bucket2vec_embeddings.weight * iid_buckets_bin
        return uid_vecs, iid_vecs

    def gumbel_form_bin(self, input_vector):
        # input_vector ? * v
        input_vector_point = torch.stack(
            (input_vector, -input_vector + 1), dim=-1)  # ? * v * 2
        input_vector_point = self.gumbel_softmax(
            input_vector_point)  # out * in * 2
        input_vector_point = input_vector_point[..., 0]  # out * in
        input_vector_all = self.gumbel_softmax(input_vector)  # out * in
        output = -(-input_vector_all + 1) * (-input_vector_all + 1) + 1
        return output

    def forward(self, batch, *args, **kwargs):
        LAYER_NORM = True
        u_ids = batch[UID]  # B * 1
        i_ids = batch[IID]  # B * S
        if self.bucket_size > 0:
            u_vectors, i_vectors = self.form_id_embedding(u_ids, i_ids)
        else:
            u_vectors = self.uid_embeddings(u_ids).view(
                [u_ids.size(0), u_ids.size(1), self.latent_dim, self.vec_size])  # B * 1 * la * v
            i_vectors = self.iid_embeddings(i_ids).view(
                [i_ids.size(0), i_ids.size(1), self.latent_dim, self.vec_size])  # B * S * la * v
        mh_features = self.concat_features(batch=batch)  # B * S * mh
        u_vectors = u_vectors.expand_as(i_vectors)
        ui_vectors = torch.cat((u_vectors, i_vectors),
                               dim=-1).squeeze(dim=-2)  # B * S * 2v

        mh_features_vectors = self.feature_embeddings(
            mh_features)  # B * S * mh * v
        if self.use_id:
            # B * S * (2la+mhdim) * v
            features = torch.cat(
                [u_vectors, i_vectors, mh_features_vectors], dim=-2)
        else:
            features = mh_features_vectors
        # B * S * (2la+mhdim) * v
        print(features.shape)
        features = self.layer_norm(features) if LAYER_NORM else features

        for li, size in enumerate(self.layers):
            select = self.select_layers[li]()  # 2l * f
            select_features = features.unsqueeze(
                dim=-3) * select.unsqueeze(dim=-1)  # B * S * 2l * f * v
            if self.training:
                regularizer_vecs = [select_features[..., :size, :, :],
                                    select_features[..., size:, :, :]]  # 2 * (B * S * l * f * v)
                regularizer_select = [select[..., :size, :],
                                      select[..., size:, :]]  # 2 * (l * f)
            and_features = self.and_layer(
                select_features[..., :size, :, :], select[..., :size, :])  # B * S * l * v
            or_features = self.or_layer(
                select_features[..., size:, :, :], select[..., size:, :])  # B * S * l * v

            features = torch.cat((and_features, or_features),
                                 dim=-2)  # B * S * 2l * v
            features = self.layer_norm(
                features) if LAYER_NORM else features  # B * S * l * v

        prediction = self.similarity(
            features, self.true, sigmoid=False)  # B * S * l
        out_dict = {LAST_LAYER_OUTPUT: prediction}
        if self.output_strategy == 'sum':
            prediction = prediction.sum(-1)
        elif self.output_strategy == 'voting':
            weight = torch.nn.functional.softmax(self.weight, dim=-1)
            prediction = (weight*prediction).sum(-1)
        elif 'adaptive' in self.output_strategy:
            if 'ui' in self.output_strategy:
                weight = self.weight_layer(ui_vectors)  # B * S * 2l
            elif 'u' in self.output_strategy:
                weight = self.weight_layer(u_vectors.squeeze(dim=-2))
            elif 'i' in self.output_strategy:
                weight = self.weight_layer(i_vectors.squeeze(dim=-2))
            
            prediction = (weight*prediction).sum(-1)
            self.TensorboardWriter.add_histogram(
                'weight distribution', weight[0, 0, :], self.train_step)
            out_dict['weight'] = weight
            self.train_step = self.train_step+1
        # out_dict[FEATURE_VALUES] = store_feature
        # prediction = self.out_layer(prediction).squeeze(dim=-1)
        out_dict[PREDICTION] = prediction
        if self.training:
            out_dict[REGULARIZER_VEC] = regularizer_vecs
            out_dict[REGULARIZER_SELECT] = regularizer_select
        return self.format_output(batch, out_dict)

    def on_test_epoch_end(self) -> None:
        if self.output_strategy == 'voting':
            self.model_logger.info(
                'layer_weight_1'+str(self.weight[0, :self.layers[-1]]))
            self.model_logger.info(
                'layer_weight_2'+str(self.weight[0, self.layers[-1]:]))
        super().on_test_epoch_end()

    def logic_regularizer(self, out_dict):
        constraint = out_dict[REGULARIZER_VEC]  # ? * (B * S * l * f * v)
        constraint_select = out_dict[REGULARIZER_SELECT]  # ? * l * f

        false = -self.true
        loss_list = torch.empty((LOSS_NUM), device=self.device)
        # # regularizer
        # length
        # r_length = constraint.view(-1, self.vec_size).norm(dim=-1)
        # r_length = r_length.sum() if self.loss_sum == 1 else r_length.mean()
        # loss_list[0] = r_length

        # # not
        # r_not_true = self.similarity(false, self.true)
        # r_not_true = r_not_true.sum() if self.loss_sum == 1 else r_not_true.mean()
        # loss_list[1] = r_not_true

        # r_not_self = self.similarity(
        #     self.neg_layer(constraint), constraint).sum()
        # r_not_self = r_not_self.sum() if self.loss_sum == 1 else r_not_self.mean()
        # loss_list[2] = r_not_self

        # r_not_not_self = - \
        #     self.similarity(self.neg_layer(
        #         self.neg_layer(constraint)), constraint) + 1
        # r_not_not_self = r_not_not_self.sum() if self.loss_sum == 1 else r_not_not_self.mean()
        # loss_list[3] = r_not_not_self
        r_length = []
        r_and_true = []
        r_and_false = []
        r_or_true = []
        r_or_false = []
        r_and_self = []
        r_and_not_self = []
        r_or_self = []
        r_or_not_self = []

        for i in range(0, len(constraint_select), 2):
            reg_vec_and = constraint[i]  # B * S * l * f * v
            reg_select_and = constraint_select[i]  # l * f
            B, S, l, f, v = reg_vec_and.shape

            expand_true = self.true.expand(B, S, l, 1, v)  # B * S * l * 1 * v
            expand_false = false.expand(B, S, l, 1, v)  # B * S * l * 1 * v
            expand_mask = torch.ones(l*1).to(self.device)  # l * 1

            # reg_select_and_add = torch.cat(
            #     (reg_select_and, expand_mask), dim=-1)  # l * (f+1)
            reg_vec_and_add_true = torch.cat(
                (reg_vec_and, expand_true), dim=-2)  # B * S * l * (f+1) * v
            reg_vec_and_add_false = torch.cat(
                (reg_vec_and, expand_false), dim=-2)

            # lenth
            r_length.append(reg_vec_and.norm(dim=-1))  # each: B * S * l * f

            # x and true = x
            r_and_true.append(
                1-self.similarity(self.and_layer(reg_vec_and), self.and_layer(reg_vec_and_add_true)))
            # X and false, result of and_layer should be B * S * l * v
            r_and_false.append(
                1-self.similarity(self.and_layer(reg_vec_and_add_false), false.expand(B, S, l, v)))

            # r_and_self = r, r and not self = false
            sample_index_and = torch.randint(f, size=(
                B, S, l), device=self.device).view(B, S, l, 1, 1).expand(B, S, l, 1, v)  # B * S * l * 1 * v
            sample_self_and = torch.gather(
                reg_vec_and, 3, sample_index_and)  # B * S * l * 1 * v
            sample_not_self_and = -sample_self_and  # B * S * l * 1 * v

            reg_vec_and_add_self = torch.cat(
                (reg_vec_and, sample_self_and), dim=-2)  # B * S * l * (f+1) * v
            reg_vec_and_add_not_self = torch.cat(
                (reg_vec_and, sample_not_self_and), dim=-2)

            r_and_self.append(
                1-self.similarity(self.and_layer(reg_vec_and_add_self), self.and_layer(reg_vec_and)))

            r_and_not_self.append(
                1-self.similarity(self.and_layer(reg_vec_and_add_not_self), false.expand(B, S, l, v)))

            # begin regularizer for or
            # r or true = true, r or false = r
            reg_vec_or = constraint[i+1]
            reg_select_or = constraint_select[i+1]

            # r_length

            r_length.append(reg_vec_or.norm(dim=-1))

            # reg_select_or_add = torch.cat((reg_select_or, expand_mask), dim=-1)
            reg_vec_or_add_true = torch.cat((reg_vec_or, expand_true), dim=-2)
            reg_vec_or_add_false = torch.cat(
                (reg_vec_or, expand_false), dim=-2)

            r_or_true.append(
                1-self.similarity(self.or_layer(reg_vec_or_add_true), self.true.expand(B, S, l, v)))
            r_or_false.append(1-self.similarity(self.or_layer(reg_vec_or_add_false),
                                                self.or_layer(reg_vec_or, reg_select_or)))

            # r_or_self = r, r or not self = true
            sample_index_or = torch.randint(f, size=(
                B, S, l), device=self.device).view(B, S, l, 1, 1).expand(B, S, l, 1, v)  # B * S * l * 1 * v
            sample_self_or = torch.gather(
                reg_vec_or, 3, sample_index_or)  # B * S * l * 1 * v
            sample_not_self_or = -sample_self_or  # B * S * l * 1 * v

            reg_vec_or_add_self = torch.cat(
                (reg_vec_or, sample_self_or), dim=-2)  # B * S * l * (f+1) * v
            reg_vec_or_add_not_self = torch.cat(
                (reg_vec_or, sample_not_self_or), dim=-2)

            r_or_self.append(
                1-self.similarity(self.or_layer(reg_vec_or_add_self), self.or_layer(reg_vec_or)))
            r_or_not_self.append(
                1-self.similarity(self.or_layer(reg_vec_or_add_not_self), self.true.expand(B, S, l, v)))

        # length
        r_length = torch.stack(r_length).sum(
        ) if self.loss_sum == 1 else torch.stack(r_length).mean()
        loss_list[0] = r_length
        # and
        r_and_true = torch.stack(r_and_true).sum(
        ) if self.loss_sum == 1 else torch.stack(r_and_true).mean()
        loss_list[1] = r_and_true
        r_and_false = torch.stack(r_and_false).sum(
        ) if self.loss_sum == 1 else torch.stack(r_and_false).mean()
        loss_list[2] = r_and_false
        r_and_self = torch.stack(r_and_self).sum(
        ) if self.loss_sum == 1 else torch.stack(r_and_false).mean()
        loss_list[3] = r_and_self
        r_and_not_self = torch.stack(r_and_not_self).sum(
        ) if self.loss_sum == 1 else torch.stack(r_and_false).mean()
        loss_list[4] = r_and_not_self

        # or
        r_or_true = torch.stack(r_or_true).sum(
        ) if self.loss_sum == 1 else torch.stack(r_or_true).mean()
        loss_list[5] = r_or_true
        r_or_false = torch.stack(r_or_false).sum(
        ) if self.loss_sum == 1 else torch.stack(r_or_false).mean()
        loss_list[6] = r_or_false
        r_or_self = torch.stack(r_or_self).sum(
        ) if self.loss_sum == 1 else torch.stack(r_or_self).mean()
        loss_list[7] = r_or_self
        r_or_not_self = torch.stack(r_or_not_self).sum(
        ) if self.loss_sum == 1 else torch.stack(r_or_not_self).mean()
        loss_list[8] = r_or_not_self

        if self.adaptive_loss:
            r_loss = torch.sum(torch.mul(1/torch.pow(self.loss_weights, 2), loss_list)
                               )+torch.sum(torch.log(torch.pow(self.loss_weights, 2)))
        else:
            r_loss = r_and_true + r_and_false + r_and_self + r_and_not_self + \
                r_or_true + r_or_false + r_or_self + r_or_not_self

        out_dict[LOGIC_REGULARIZER] = r_loss

        return out_dict
    
    def mask_loss(self):
        mask_loss = 0
        for li, size in enumerate(self.layers):
            mask_loss += self.select_layers[li]().sum()
        return mask_loss

    def loss_func(self, batch, out_dict, *args, **kwargs):
        prediction, label = out_dict[PREDICTION], out_dict[LABEL]
        loss = super().loss_func(batch, {PREDICTION: prediction, LABEL: label})

        if self.r_logic > 0:
            r_loss = self.logic_regularizer(out_dict)[LOGIC_REGULARIZER]
            loss += r_loss * self.r_logic
        if self.r_mask > 0:
            loss += self.r_mask*self.mask_loss()
        # if self.r_length > 0:
        #     r_length = out_dict[REGULARIZER_VEC].norm(dim=-1).sum()
        #     loss += r_length * self.r_length
        return loss

    def mask_analysis(self):
        print("Do Explanation Analysis")
        N_feature = self.select_layers[0]().shape[1]
        edge_num = list()
        print("N_feature:", N_feature)
        for li, size in enumerate(self.layers):
            for i in range(size*2):
                if i < size:
                    print(
                        "the AND NODE {0} in Layer {1} is connected with:".format(i, li))
                else:
                    print(
                        "the OR NODE {0} in Layer {1} is connected with:".format(i, li))
                mask = self.select_layers[li]()[i]
                print(mask)
                position = torch.where(mask == 1)
                print(list(position))
                edge_num.append(list(position)[0].shape[0])
        print(edge_num)
        self.model_logger.info(
            "average edge num:{0}\n".format(np.mean(np.array(edge_num))))

