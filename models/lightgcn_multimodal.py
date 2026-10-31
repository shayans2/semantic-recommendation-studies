import torch as t
from torch import nn
from loss_funcs import cal_bpr_loss, reg_params, cal_infonce_loss
from utils import SpAdjEdgeDrop, spmm

init = nn.init.xavier_uniform_


class LightGCN(nn.Module):
    def __init__(self, data_handler, embedding_size, keep_rate, layer_num, reg_weight, kd_weight, kd_temperature):
        super(LightGCN, self).__init__()
        # Configuration parameters
        self.user_num = data_handler.user_num  # Number of users in the dataset
        self.item_num = data_handler.item_num  # Number of items in the dataset
        self.embedding_size = embedding_size  # Embedding size
        self.keep_rate = keep_rate  # Dropout keep rate for edges
        self.layer_num = layer_num  # Number of layers in GCN
        self.reg_weight = reg_weight  # Regularization weight
        self.cl_weight = 1.0e-1
        self.cl_temperature = 0.2
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.user_sem_emb = data_handler.user_sem_emb
        self.item_sem_emb = data_handler.item_sem_emb

        # Load adjacency matrix and initialize embeddings
        self.adj = data_handler.torch_adj
        self.user_embeds = nn.Parameter(
            init(t.empty(self.user_num, self.embedding_size))
        )
        self.item_embeds = nn.Parameter(
            init(t.empty(self.item_num, self.embedding_size))
        )

        # Edge dropper initialization for training with dropout
        self.edge_dropper = SpAdjEdgeDrop()

        device = data_handler.device
        self.usrprf_embeds = t.tensor(self.user_sem_emb).float().to(device)
        self.itmprf_embeds = t.tensor(self.item_sem_emb).float().to(device)

        self.mlp = nn.Sequential(
            nn.Linear(
                self.usrprf_embeds.shape[1],
                (self.usrprf_embeds.shape[1] + self.embedding_size) // 2,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                (self.usrprf_embeds.shape[1] + self.embedding_size) // 2,
                self.embedding_size,
            ),
        )

        # Final embeddings and training state
        self.final_embeds = None
        self.is_training = False

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _propagate(self, adj, embeds):
        """Sparse matrix multiplication for GCN layer propagation"""
        return spmm(adj, embeds)

    def reset_cache(self):
        """Drop the cached propagation.

        final_embeds is a plain attribute, so it does NOT travel in state_dict.
        Any change to the parameters (an optimizer step, or load_state_dict)
        must be followed by this call, or evaluation keeps serving embeddings
        that belong to a different set of weights.
        """
        self.final_embeds = None
        self.is_training = False

    def forward(self, adj=None, keep_rate=1.0):
        """Forward pass for LightGCN"""
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

        # Propagate embeddings across layers
        for _ in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        # Sum up embeddings from all layers (NOW CHANGED TO AVG)
        # embeds = sum(embeds_list)
        embeds = sum(embeds_list) / len(embeds_list)
        if not self.is_training:
            # Only cache evaluation-time embeddings. The training pass runs
            # with edge dropout and must never be reused for prediction.
            self.final_embeds = embeds
        return embeds[: self.user_num], embeds[self.user_num :]

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        """Calculate the BPR loss and regularization loss"""
        self.is_training = True
        user_embeds1, item_embeds1 = self.forward(self.adj, self.keep_rate)
        user_embeds2, item_embeds2 = self.forward(self.adj, self.keep_rate)
        user_embeds3, item_embeds3 = self.forward(self.adj, 1.0)

        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(
            user_embeds1, item_embeds1, batch_data
        )
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(
            user_embeds2, item_embeds2, batch_data
        )
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(
            user_embeds3, item_embeds3, batch_data
        )

        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(
            usrprf_embeds, itmprf_embeds, batch_data
        )

        bpr_loss = (
            cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]
        )

        cl_loss = (
            cal_infonce_loss(
                anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature
            )
            + cal_infonce_loss(
                pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature
            )
            + cal_infonce_loss(
                neg_embeds1, neg_embeds2, item_embeds2, self.cl_temperature
            )
        )
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight

        kd_loss = (
            cal_infonce_loss(
                anc_embeds3, ancprf_embeds, usrprf_embeds, self.kd_temperature
            )
            + cal_infonce_loss(
                pos_embeds3, posprf_embeds, posprf_embeds, self.kd_temperature
            )
            + cal_infonce_loss(
                neg_embeds3, negprf_embeds, negprf_embeds, self.kd_temperature
            )
        )
        kd_loss /= anc_embeds3.shape[0]
        kd_loss *= self.kd_weight

        reg_loss = self.reg_weight * reg_params(self)

        loss = bpr_loss + reg_loss + cl_loss + kd_loss
        losses = {
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "cl_loss": cl_loss,
            "kd_loss": kd_loss,
        }
        return loss, losses

    def _mask_predict(self, full_preds, train_mask):
        """Apply mask to predictions, hiding training data"""
        return full_preds * (1 - train_mask) - 1e8 * train_mask

    def full_predict(self, batch_data):
        """Full prediction for evaluation"""
        self.is_training = False
        user_embeds, item_embeds = self.forward(self.adj, 1.0)

        # Unpack the batch data
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        # Calculate predictions and mask out training pairs
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)

        return full_preds
