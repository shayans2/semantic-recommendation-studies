import torch as t
from torch import nn
import torch.nn.functional as F
from loss_funcs import cal_bpr_loss, reg_params, cal_infonce_loss
from utils import SpAdjEdgeDrop, spmm

init = nn.init.xavier_uniform_


class LightGCN_plus(nn.Module):
    def __init__(
        self,
        data_handler,
        embedding_size,
        keep_rate,
        layer_num,
        reg_weight,
        kd_weight,
        kd_temperature,
    ):
        super(LightGCN_plus, self).__init__()
        # Configuration parameters
        self.user_num = data_handler.user_num  # Number of users in the dataset
        self.item_num = data_handler.item_num  # Number of items in the dataset
        self.embedding_size = (
            embedding_size  # Set embedding size (can be adjusted as needed)
        )
        self.keep_rate = keep_rate  # Dropout keep rate for edges
        self.layer_num = layer_num  # Number of GCN layers
        self.reg_weight = reg_weight  # Regularization weight
        self.kd_weight = kd_weight  # Knowledge distillation weight
        self.kd_temperature = kd_temperature  # Temperature for distillation loss
        self.user_sem_emb = data_handler.user_sem_emb
        self.item_sem_emb = data_handler.item_sem_emb

        self.user_sim_emb = data_handler.user_sim_emb
        self.item_sim_emb = data_handler.item_sim_emb

        # Load adjacency matrix and initialize embeddings
        self.adj = data_handler.torch_adj
        self.user_embeds = nn.Parameter(
            init(t.empty(self.user_num, self.embedding_size))
        )
        self.item_embeds = nn.Parameter(
            init(t.empty(self.item_num, self.embedding_size))
        )

        # Edge dropper initialization
        self.edge_dropper = SpAdjEdgeDrop()

        # Semantic and similarity embeddings
        device = data_handler.device
        self.user_sem_emb = t.tensor(self.user_sem_emb).float().to(device)
        self.user_sim_emb = t.tensor(self.user_sim_emb).float().to(device)
        self.item_sem_emb = t.tensor(self.item_sem_emb).float().to(device)
        self.item_sim_emb = t.tensor(self.item_sim_emb).float().to(device)

        # Separate MLPs for semantic and similarity embeddings
        proj_out_dim = self.embedding_size // 2
        self.user_sem_proj = nn.Linear(self.user_sem_emb.shape[1], proj_out_dim)
        self.user_sim_proj = nn.Linear(self.user_sim_emb.shape[1], proj_out_dim)
        self.item_sem_proj = nn.Linear(self.item_sem_emb.shape[1], proj_out_dim)
        self.item_sim_proj = nn.Linear(self.item_sim_emb.shape[1], proj_out_dim)

        # Added normalization layers right after the projections
        self.user_sem_norm = nn.LayerNorm(proj_out_dim)
        self.user_sim_norm = nn.LayerNorm(proj_out_dim)
        self.item_sem_norm = nn.LayerNorm(proj_out_dim)
        self.item_sim_norm = nn.LayerNorm(proj_out_dim)

        self.final_embeds = None
        self.is_training = False

        self._init_weight()

    def _init_weight(self):
        """Initialize weights for the linear layers."""
        for layer in [
            self.user_sem_proj,
            self.user_sim_proj,
            self.item_sem_proj,
            self.item_sim_proj,
        ]:
            init(layer.weight)

    def _propagate(self, adj, embeds):
        """Sparse matrix multiplication for GCN layer propagation."""
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
        """Forward pass for LightGCN_plus."""
        if adj is None:
            adj = self.adj

        if not self.is_training and self.final_embeds is not None:
            return (
                self.final_embeds[: self.user_num],
                self.final_embeds[self.user_num :],
            )

        # Concatenate CF embeddings for users and items
        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]

        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)

        # Propagate embeddings across layers
        for _ in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        # Average the embeddings from each layer
        embeds = sum(embeds_list) / len(embeds_list)
        # Normalize the CF embeddings using L2 normalization
        embeds = F.normalize(embeds, p=2, dim=1)
        if not self.is_training:
            # Only cache evaluation-time embeddings. The training pass runs
            # with edge dropout and must never be reused for prediction.
            self.final_embeds = embeds
        return embeds[: self.user_num], embeds[self.user_num :]

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        """Extract embeddings for the given batch data."""
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        """Calculate the BPR loss, regularization loss, and knowledge distillation loss."""
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)

        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(
            user_embeds, item_embeds, batch_data
        )

        # Project semantic and similarity embeddings and apply normalization
        user_sem_proj = self.user_sem_proj(self.user_sem_emb)
        user_sem_proj = self.user_sem_norm(user_sem_proj)
        user_sim_proj = self.user_sim_proj(self.user_sim_emb)
        user_sim_proj = self.user_sim_norm(user_sim_proj)
        item_sem_proj = self.item_sem_proj(self.item_sem_emb)
        item_sem_proj = self.item_sem_norm(item_sem_proj)
        item_sim_proj = self.item_sim_proj(self.item_sim_emb)
        item_sim_proj = self.item_sim_norm(item_sim_proj)

        usrprf_embeds = t.cat([user_sem_proj, user_sim_proj], dim=1)
        itmprf_embeds = t.cat([item_sem_proj, item_sim_proj], dim=1)

        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(
            usrprf_embeds, itmprf_embeds, batch_data
        )

        # Calculate BPR loss
        bpr_loss = (
            cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        )
        reg_loss = self.reg_weight * reg_params(self)

        # Calculate knowledge distillation loss
        kd_loss = (
            cal_infonce_loss(
                anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature
            )
            + cal_infonce_loss(
                pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature
            )
            + cal_infonce_loss(
                neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature
            )
        )
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        # Total loss
        loss = bpr_loss + reg_loss + kd_loss
        losses = {"bpr_loss": bpr_loss, "reg_loss": reg_loss, "kd_loss": kd_loss}
        return loss, losses

    def _mask_predict(self, full_preds, train_mask):
        """Apply a mask to predictions, hiding training data."""
        return full_preds * (1 - train_mask) - 1e8 * train_mask

    def full_predict(self, batch_data):
        """Full prediction for evaluation."""
        self.is_training = False
        user_embeds, item_embeds = self.forward(self.adj, 1.0)

        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)

        return full_preds
