"""LightGCN_plus, faithful to the RLMRec reference implementation.

Adapted from HKUDS/RLMRec at commit
22413752246de3dee8ab0d509f7f7a8889080f95 under Apache-2.0. This file has
been modified for device portability, cache correctness, and integration with
this repository. See LICENSES/RLMRec-Apache-2.0.txt and
THIRD_PARTY_NOTICES.md.

`models/lightgcn_plus.py` in this repo diverges from the published RLMRec model
in ways that change its behaviour, so it is not a sound baseline to compare
against. This module reproduces the reference so comparisons are fair.

Differences from `models/lightgcn_plus.py` (i.e. what this restores):
  - Layer aggregation is `sum(embeds_list)`, not `sum(...) / len(...)`.
    At layer_num=3 the mean is ~4x smaller, which shrinks BPR's dot products
    ~16x while the InfoNCE anchor stays scale-invariant (it L2-normalises), so
    the anchor overwhelms the ranking signal.
  - L2 regularisation is the sum of SQUARED norms (`reg_params_squared`),
    matching RLMRec's `loss_utils.reg_params`.

Deliberate deviations from the reference, each for a reason:
  - `.cuda()` is replaced by the DataHandler's device, so this runs on CUDA,
    Apple Metal, or CPU. Propagation goes through `utils.spmm`, which is
    equivalent to `t.spmm` but has an MPS-compatible path.
  - The reference's `full_predict` sets `is_training = False` AFTER calling
    `forward`, and `forward` caches unconditionally. That combination serves a
    stale cache after `load_state_dict`, so evaluation reports the wrong
    checkpoint's numbers. This is a latent bug in the reference, not a design
    choice, and reproducing it would handicap this baseline against models that
    do not have it. Fixed here: `is_training` is cleared before propagating,
    caching happens only outside training, and `reset_cache()` is provided.

Everything else -- the MLP shape and init, the three-term InfoNCE distillation
with its asymmetric denominators (full user set for the anchor, in-batch for
positives and negatives), the loss weighting, and edge dropout -- is as
published.
"""

import torch as t
from torch import nn

from loss_funcs import cal_bpr_loss, cal_infonce_loss, reg_params_squared
from utils import SpAdjEdgeDrop, spmm

init = nn.init.xavier_uniform_


class LightGCN_plus_RLMRec(nn.Module):
    def __init__(self, data_handler, embedding_size, keep_rate, layer_num,
                 reg_weight, kd_weight, kd_temperature):
        super(LightGCN_plus_RLMRec, self).__init__()
        self.user_num = data_handler.user_num
        self.item_num = data_handler.item_num
        self.embedding_size = embedding_size
        self.keep_rate = keep_rate
        self.layer_num = layer_num
        self.reg_weight = reg_weight
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature

        self.adj = data_handler.torch_adj
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()

        device = data_handler.device
        if data_handler.user_sem_emb is None or data_handler.item_sem_emb is None:
            raise ValueError(
                "LightGCN_plus_RLMRec requires user and item semantic embeddings; "
                "usr_emb_np.pkl / itm_emb_np.pkl were not found for this dataset."
            )
        self.usrprf_embeds = t.tensor(data_handler.user_sem_emb).float().to(device)
        self.itmprf_embeds = t.tensor(data_handler.item_sem_emb).float().to(device)

        prf_dim = self.usrprf_embeds.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(prf_dim, (prf_dim + embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((prf_dim + embedding_size) // 2, embedding_size),
        )

        self.final_embeds = None
        self.is_training = False

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def reset_cache(self):
        self.final_embeds = None
        self.is_training = False

    def _propagate(self, adj, embeds):
        return spmm(adj, embeds)

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return (
                self.final_embeds[: self.user_num],
                self.final_embeds[self.user_num :],
            )

        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for _ in range(self.layer_num):
            embeds_list.append(self._propagate(adj, embeds_list[-1]))

        embeds = sum(embeds_list)  # reference: sum over layers, not mean
        if not self.is_training:
            self.final_embeds = embeds
        return embeds[: self.user_num], embeds[self.user_num :]

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        return user_embeds[ancs], item_embeds[poss], item_embeds[negs]

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)

        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(
            user_embeds, item_embeds, batch_data
        )

        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(
            usrprf_embeds, itmprf_embeds, batch_data
        )

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params_squared(self)

        kd_loss = (
            cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature)
            + cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature)
            + cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        )
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        loss = bpr_loss + reg_loss + kd_loss
        return loss, {"bpr_loss": bpr_loss, "reg_loss": reg_loss, "kd_loss": kd_loss}

    def _mask_predict(self, full_preds, train_mask):
        return full_preds * (1 - train_mask) - 1e8 * train_mask

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        full_preds = user_embeds[pck_users.long()] @ item_embeds.T
        return self._mask_predict(full_preds, train_mask)
