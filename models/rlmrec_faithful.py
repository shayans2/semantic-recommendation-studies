"""Exact reproductions of RLMRec's released models.

Adapted from HKUDS/RLMRec at commit
22413752246de3dee8ab0d509f7f7a8889080f95 under Apache-2.0. This file has
been modified for device portability, explicit ablations, cache correctness,
and integration with this repository. See LICENSES/RLMRec-Apache-2.0.txt and
THIRD_PARTY_NOTICES.md.

Verified line-by-line against https://github.com/HKUDS/RLMRec :

  encoder/models/general_cf/lightgcn.py       embeds = sum(embeds_list)
  encoder/models/general_cf/lightgcn_plus.py  same aggregation + InfoNCE anchor
  encoder/models/model_utils.py               newVals = vals[mask]# / keep_rate
  encoder/models/loss_utils.py                reg_loss += W.norm(2).square()
  encoder/config/modelconf/lightgcn_plus.yml  embedding_size 32, batch_size 4096

Differences from models/lightgcn.py and models/lightgcn_plus_rlmrec.py in this
repository, all of which matter for comparability with the published numbers:

  - Sum aggregation over layers, not mean. This repository's LightGCN averages,
    which is the original LightGCN paper's formulation but not RLMRec's.
  - Edge dropout does NOT rescale survivors. RLMRec commented that division out.
  - embedding_size 32 and batch_size 4096 come from their config, against this
    repository's 64 and 1024.

The one deliberate deviation is the cache-ordering defect: RLMRec's full_predict
clears is_training AFTER propagating and caches unconditionally, so a reloaded
checkpoint is ignored. Reproducing that would mean evaluating the wrong weights.
It is fixed here, which can only help these models.
"""

import torch as t
from torch import nn

from loss_funcs import cal_bpr_loss, cal_infonce_loss, reg_params_squared
from utils import SpAdjEdgeDrop, spmm

init = nn.init.xavier_uniform_


class _Base(nn.Module):
    def __init__(self, data_handler, embedding_size, keep_rate, layer_num, reg_weight):
        super().__init__()
        self.user_num = data_handler.user_num
        self.item_num = data_handler.item_num
        self.embedding_size = embedding_size
        self.keep_rate = keep_rate
        self.layer_num = layer_num
        self.reg_weight = reg_weight
        self.adj = data_handler.torch_adj
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop(rescale=False)   # RLMRec: no rescale
        self.final_embeds = None
        self.is_training = False

    def reset_cache(self):
        self.final_embeds = None
        self.is_training = False

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return (self.final_embeds[: self.user_num],
                    self.final_embeds[self.user_num:])
        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for _ in range(self.layer_num):
            embeds_list.append(spmm(adj, embeds_list[-1]))
        embeds = sum(embeds_list)          # RLMRec: sum, not mean
        if not self.is_training:
            self.final_embeds = embeds
        return embeds[: self.user_num], embeds[self.user_num:]

    def _pick(self, ue, ie, batch_data):
        ancs, poss, negs = batch_data
        return ue[ancs], ie[poss], ie[negs]

    def _mask_predict(self, full_preds, train_mask):
        return full_preds * (1 - train_mask) - 1e8 * train_mask

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        pck_users, train_mask = batch_data
        return self._mask_predict(
            user_embeds[pck_users.long()] @ item_embeds.T, train_mask)


class LightGCN_RLMRec(_Base):
    """RLMRec's LightGCN baseline, exactly as released."""

    def cal_loss(self, batch_data):
        self.is_training = True
        ue, ie = self.forward(self.adj, self.keep_rate)
        a, p, n = self._pick(ue, ie, batch_data)
        bpr = cal_bpr_loss(a, p, n) / a.shape[0]
        reg = self.reg_weight * reg_params_squared(self)
        return bpr + reg, {"bpr_loss": bpr, "reg_loss": reg}


class LightGCN_plus_RLMRec_Exact(_Base):
    """RLMRec's LightGCN_plus, exactly as released."""

    def __init__(self, data_handler, embedding_size, keep_rate, layer_num,
                 reg_weight, kd_weight, kd_temperature):
        super().__init__(data_handler, embedding_size, keep_rate, layer_num, reg_weight)
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        device = data_handler.device
        if data_handler.user_sem_emb is None or data_handler.item_sem_emb is None:
            raise ValueError("needs usr_emb_np.pkl and itm_emb_np.pkl")
        self.usrprf_embeds = t.tensor(data_handler.user_sem_emb).float().to(device)
        self.itmprf_embeds = t.tensor(data_handler.item_sem_emb).float().to(device)
        d = self.usrprf_embeds.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(d, (d + embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((d + embedding_size) // 2, embedding_size),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def cal_loss(self, batch_data):
        self.is_training = True
        ue, ie = self.forward(self.adj, self.keep_rate)
        a, p, n = self._pick(ue, ie, batch_data)
        up = self.mlp(self.usrprf_embeds)
        ip = self.mlp(self.itmprf_embeds)
        ap, pp, np_ = self._pick(up, ip, batch_data)
        bpr = cal_bpr_loss(a, p, n) / a.shape[0]
        reg = self.reg_weight * reg_params_squared(self)
        kd = (cal_infonce_loss(a, ap, up, self.kd_temperature)
              + cal_infonce_loss(p, pp, pp, self.kd_temperature)
              + cal_infonce_loss(n, np_, np_, self.kd_temperature))
        kd = kd / a.shape[0] * self.kd_weight
        return bpr + reg + kd, {"bpr_loss": bpr, "reg_loss": reg, "kd_loss": kd}


class LightGCN_plus_RLMRec_ItemOnly(_Base):
    """Controlled RLMRec ablation that removes user semantic profiles.

    Everything inherited from the released RLMRec implementation is kept
    unchanged: sum aggregation, unscaled edge dropout, squared-L2
    regularisation, and the shared semantic projector.  The only change is to
    delete the user-profile InfoNCE term.  Positive- and negative-item terms
    remain, so this answers whether user profiles explain RLMRec's behaviour
    without turning the model into a different item-side objective.

    ``kd_weight`` deliberately retains the same meaning as in full RLMRec.
    Retuning it is a separate experiment; silently multiplying it by 3/2 here
    would confound profile removal with loss reweighting.
    """

    def __init__(self, data_handler, embedding_size, keep_rate, layer_num,
                 reg_weight, kd_weight, kd_temperature):
        super().__init__(data_handler, embedding_size, keep_rate, layer_num, reg_weight)
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        device = data_handler.device
        if data_handler.item_sem_emb is None:
            raise ValueError("needs itm_emb_np.pkl; user profiles are not used")
        self.itmprf_embeds = t.tensor(data_handler.item_sem_emb).float().to(device)
        d = self.itmprf_embeds.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(d, (d + embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((d + embedding_size) // 2, embedding_size),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def cal_loss(self, batch_data):
        self.is_training = True
        ue, ie = self.forward(self.adj, self.keep_rate)
        a, p, n = self._pick(ue, ie, batch_data)
        ip = self.mlp(self.itmprf_embeds)
        _, pp, np_ = self._pick(ip, ip, batch_data)
        bpr = cal_bpr_loss(a, p, n) / a.shape[0]
        reg = self.reg_weight * reg_params_squared(self)
        kd = (cal_infonce_loss(p, pp, pp, self.kd_temperature)
              + cal_infonce_loss(n, np_, np_, self.kd_temperature))
        kd = kd / a.shape[0] * self.kd_weight
        return bpr + reg + kd, {"bpr_loss": bpr, "reg_loss": reg, "kd_loss": kd}
