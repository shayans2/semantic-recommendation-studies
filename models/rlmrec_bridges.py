"""Controlled bridges from item-only RLMRec to FREEDOM without its item graph.

Every one-factor subclass starts from the faithful item-only RLMRec model and
changes exactly one training choice.  The combined subclass applies all three
code-level choices; its KD weight is supplied by configuration because that is
an independent scalar comparison.
"""

import torch as t

from loss_funcs import cal_bpr_loss, cal_infonce_loss, reg_params_squared
from models.rlmrec_faithful import LightGCN_plus_RLMRec_ItemOnly
from utils import degree_sensitive_prune, spmm


class _ItemOnlyRLMRecBridge(LightGCN_plus_RLMRec_ItemOnly):
    mean_aggregation = False
    positive_only = False

    def __init__(self, data_handler, embedding_size, keep_rate, layer_num,
                 reg_weight, kd_weight, kd_temperature):
        super().__init__(
            data_handler, embedding_size, keep_rate, layer_num, reg_weight,
            kd_weight, kd_temperature,
        )

    def _training_adj(self, adj, keep_rate):
        return self.edge_dropper(adj, keep_rate)

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return (
                self.final_embeds[: self.user_num],
                self.final_embeds[self.user_num:],
            )

        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self._training_adj(adj, keep_rate)
        for _ in range(self.layer_num):
            embeds_list.append(spmm(adj, embeds_list[-1]))
        embeds = sum(embeds_list)
        if self.mean_aggregation:
            embeds = embeds / len(embeds_list)
        if not self.is_training:
            self.final_embeds = embeds
        return embeds[: self.user_num], embeds[self.user_num:]

    def cal_loss(self, batch_data):
        self.is_training = True
        ue, ie = self.forward(self.adj, self.keep_rate)
        a, p, n = self._pick(ue, ie, batch_data)
        ip = self.mlp(self.itmprf_embeds)
        _, pp, np_ = self._pick(ip, ip, batch_data)

        bpr = cal_bpr_loss(a, p, n) / a.shape[0]
        reg = self.reg_weight * reg_params_squared(self)
        kd = cal_infonce_loss(p, pp, pp, self.kd_temperature)
        if not self.positive_only:
            kd = kd + cal_infonce_loss(n, np_, np_, self.kd_temperature)
        kd = kd / a.shape[0] * self.kd_weight
        return bpr + reg + kd, {
            "bpr_loss": bpr, "reg_loss": reg, "kd_loss": kd,
        }


class _DegreePrunedItemOnlyRLMRecBridge(_ItemOnlyRLMRecBridge):
    """Bridge base for the conditions that genuinely resample each epoch."""

    resamples_graph_each_epoch = True

    def __init__(self, data_handler, embedding_size, keep_rate, layer_num,
                 reg_weight, kd_weight, kd_temperature, rho):
        super().__init__(
            data_handler, embedding_size, keep_rate, layer_num, reg_weight,
            kd_weight, kd_temperature,
        )
        self.rho = float(rho)
        self.trn_mat = data_handler.trn_mat
        self.device = data_handler.device
        # Do not advance the global RNG: the subsequent DataLoader shuffle
        # must match the faithful baseline for the same experiment seed.
        self.generator = t.Generator().manual_seed(t.initial_seed() + 205759)
        self.train_adj = None
        self.on_epoch_start()

    def on_epoch_start(self):
        self.train_adj = degree_sensitive_prune(
            self.trn_mat, self.rho, self.device, self.generator
        )
        self.reset_cache()

    def _training_adj(self, adj, keep_rate):
        return self.train_adj


class RLMRecItemOnlyPositiveOnly(_ItemOnlyRLMRecBridge):
    """Remove only the sampled-negative item InfoNCE term."""

    positive_only = True


class RLMRecItemOnlyMeanAggregation(_ItemOnlyRLMRecBridge):
    """Change only layer aggregation from sum to arithmetic mean."""

    mean_aggregation = True


class RLMRecItemOnlyDegreePrune(_DegreePrunedItemOnlyRLMRecBridge):
    """Replace per-batch uniform edge dropout with per-epoch degree pruning."""

    degree_prune = True


class RLMRecItemOnlyCombinedBridge(_DegreePrunedItemOnlyRLMRecBridge):
    """FREEDOM's no-structural training recipe on the RLMRec implementation."""

    mean_aggregation = True
    positive_only = True
    degree_prune = True
