import torch as t
from torch import nn
from loss_funcs import cal_bpr_loss, reg_params
from utils import SpAdjEdgeDrop, spmm

init = nn.init.xavier_uniform_


class LightGCN(nn.Module):
    def __init__(self, data_handler, embedding_size, keep_rate, layer_num, reg_weight):
        super(LightGCN, self).__init__()
        # Configuration parameters
        self.user_num = data_handler.user_num  # Number of users in the dataset
        self.item_num = data_handler.item_num  # Number of items in the dataset
        self.embedding_size = embedding_size  # Embedding size
        self.keep_rate = keep_rate  # Dropout keep rate for edges
        self.layer_num = layer_num  # Number of layers in GCN
        self.reg_weight = reg_weight  # Regularization weight

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

        # Final embeddings and training state
        self.final_embeds = None
        self.is_training = False

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

    def cal_loss(self, batch_data):
        """Calculate the BPR loss and regularization loss"""
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)

        # Extract batch data (user, positive items, negative items)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        # Calculate BPR loss and regularization loss
        bpr_loss = (
            cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        )
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss

        # Return loss and individual loss components
        return loss, {"bpr_loss": bpr_loss, "reg_loss": reg_loss}

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
