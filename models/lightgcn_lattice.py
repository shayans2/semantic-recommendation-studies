import torch as t
from torch import nn
from loss_funcs import cal_bpr_loss, reg_params, cal_infonce_loss
from utils import SpAdjEdgeDrop, spmm

# Xavier initialization shortcut.
init = nn.init.xavier_uniform_


class LightGCN_plus_LATTICE(nn.Module):
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
        """
        Parameters:
          data_handler: an object that must provide:
              - user_num, item_num
              - torch_adj (the precomputed torch sparse adjacency matrix)
              - user_sem_emb (e.g., user profile features)
              - item_image_feats (item image features, as a NumPy array or tensor)
              - item_text_feats (item text features, as a NumPy array or tensor)
          embedding_size: dimensionality for embeddings.
          keep_rate: dropout keep rate for edge dropout.
          layer_num: number of GCN propagation layers.
          reg_weight: weight for regularization loss.
          kd_weight: weight for the contrastive (knowledge distillation) loss.
          kd_temperature: temperature used in the contrastive loss.
        """
        super(LightGCN_plus_LATTICE, self).__init__()
        # Basic configuration
        self.user_num = data_handler.user_num
        self.item_num = data_handler.item_num
        self.embedding_size = embedding_size
        self.keep_rate = keep_rate
        self.layer_num = layer_num
        self.reg_weight = reg_weight
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature

        # --- Semantic/Profile embeddings for users ---
        # (For users we still assume a “profile” is available.)
        self.user_sem_emb = data_handler.user_sem_emb  # e.g., a NumPy array
        # Convert to tensor once (will be moved to device in __init__)
        device = data_handler.device
        self.usrprf_embeds = t.tensor(self.user_sem_emb).float().to(device)

        # --- LATTICE branch for items ---
        # Instead of an “item_sem_emb” we assume two modalities are available:
        self.item_image_feats = data_handler.item_image_feats  # e.g., NumPy array
        self.item_text_feats = data_handler.item_text_feats  # e.g., NumPy array
        if self.item_image_feats is None or self.item_text_feats is None:
            raise ValueError(
                "LightGCN_plus_LATTICE needs per-item image and text features. "
                "Set data_handler.item_image_feats and data_handler.item_text_feats "
                "(each an array with item_num rows) before building this model; "
                "no dataset under ./data currently ships them."
            )

        # --- Adjacency matrix for graph propagation ---
        self.adj = data_handler.torch_adj

        # --- Collaborative embeddings (learned) for users and items ---
        self.user_embeds = nn.Parameter(
            init(t.empty(self.user_num, self.embedding_size))
        )
        self.item_embeds = nn.Parameter(
            init(t.empty(self.item_num, self.embedding_size))
        )

        # --- Edge dropout (for training) ---
        self.edge_dropper = SpAdjEdgeDrop()

        # --- MLP for transforming user profile features ---
        # This maps the raw user profile (semantic) features into the same embedding space.
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

        # --- Content projection network for items ---
        # This fuses image and text features (concatenated) into an embedding vector.
        # Determine the combined input dimension.
        # (Assuming the features are provided as NumPy arrays; if already tensors, that’s fine.)
        item_image_feats_tensor = t.tensor(self.item_image_feats)
        item_text_feats_tensor = t.tensor(self.item_text_feats)
        feat_dim_image = item_image_feats_tensor.shape[1]
        feat_dim_text = item_text_feats_tensor.shape[1]
        combined_dim = feat_dim_image + feat_dim_text
        self.content_projection = nn.Sequential(
            nn.Linear(combined_dim, (combined_dim + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((combined_dim + self.embedding_size) // 2, self.embedding_size),
        )

        self.final_embeds = None
        self.is_training = False

        self._init_weight()

    def _init_weight(self):
        # Initialize weights for the MLPs.
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.content_projection:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _propagate(self, adj, embeds):
        """Perform sparse matrix multiplication (one GCN layer)."""
        return spmm(adj, embeds)

    def compute_content_embeds(self):
        """Compute item content embeddings by fusing image and text features."""
        device = self.user_embeds.device
        # Ensure the features are tensors on the correct device.
        if not isinstance(self.item_image_feats, t.Tensor):
            item_image = t.tensor(self.item_image_feats).float().to(device)
        else:
            item_image = self.item_image_feats.float().to(device)
        if not isinstance(self.item_text_feats, t.Tensor):
            item_text = t.tensor(self.item_text_feats).float().to(device)
        else:
            item_text = self.item_text_feats.float().to(device)
        # Concatenate along feature dimension.
        combined_feats = t.cat([item_image, item_text], dim=1)
        # Project to embedding space.
        content_embeds = self.content_projection(combined_feats)
        return content_embeds

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
        """
        Forward pass:
          - Performs GCN propagation over the user–item graph.
          - Computes collaborative embeddings.
          - Computes item content embeddings from image and text features.
        Returns a tuple of (user_embeds, item_embeds, item_content_embeds).
        """
        if adj is None:
            adj = self.adj

        if not self.is_training and self.final_embeds is not None:
            # In inference, use precomputed collaborative embeddings.
            content_embeds = self.compute_content_embeds()
            return (
                self.final_embeds[: self.user_num],
                self.final_embeds[self.user_num :],
                content_embeds,
            )

        # Concatenate the user and item collaborative embeddings.
        embeds = t.cat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]

        if self.is_training:
            # Optionally apply edge dropout.
            adj = self.edge_dropper(adj, keep_rate)

        for _ in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        # Aggregate the embeddings from each layer (here, averaging).
        embeds = sum(embeds_list) / len(embeds_list)
        if not self.is_training:
            # Only cache evaluation-time embeddings. The training pass runs
            # with edge dropout and must never be reused for prediction.
            self.final_embeds = embeds

        # Compute the content embeddings for items.
        content_embeds = self.compute_content_embeds()
        return embeds[: self.user_num], embeds[self.user_num :], content_embeds

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        """
        Given a batch (tuple of indices: (user_idx, pos_item_idx, neg_item_idx)),
        select the corresponding embeddings.
        """
        anc, pos, neg = batch_data
        anc_embeds = user_embeds[anc]
        pos_embeds = item_embeds[pos]
        neg_embeds = item_embeds[neg]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        """
        Calculate the overall loss composed of:
          - BPR loss (to separate positive and negative item scores)
          - Regularization loss
          - Contrastive (InfoNCE) loss:
              * Aligns the user collaborative embedding with its MLP-transformed profile.
              * Aligns the item collaborative embedding with its content (image/text) embedding.
        """
        self.is_training = True
        # Forward pass: obtain collaborative embeddings and item content embeddings.
        user_embeds, item_embeds, content_embeds = self.forward(
            self.adj, self.keep_rate
        )
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(
            user_embeds, item_embeds, batch_data
        )

        # For users, transform profile features.
        usrprf_all = self.mlp(self.usrprf_embeds)
        ancprf_embeds = usrprf_all[batch_data[0]]
        # For items, use the content embeddings from the LATTICE branch.
        posprf_embeds = content_embeds[batch_data[1]]
        negprf_embeds = content_embeds[batch_data[2]]

        # BPR loss (recommendation loss)
        bpr_loss = (
            cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        )
        reg_loss = self.reg_weight * reg_params(self)

        # Contrastive (knowledge distillation) loss:
        kd_loss = (
            cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_all, self.kd_temperature)
            + cal_infonce_loss(
                pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature
            )
            + cal_infonce_loss(
                neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature
            )
        )
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        loss = bpr_loss + reg_loss + kd_loss
        losses = {"bpr_loss": bpr_loss, "reg_loss": reg_loss, "kd_loss": kd_loss}
        return loss, losses

    def _mask_predict(self, full_preds, train_mask):
        """Apply a mask to hide training interactions during full prediction."""
        return full_preds * (1 - train_mask) - 1e8 * train_mask

    def full_predict(self, batch_data):
        """
        For evaluation, returns full prediction scores.
          - batch_data is assumed to be a tuple of (picked_users, train_mask)
        """
        self.is_training = False
        user_embeds, item_embeds, _ = self.forward(self.adj, 1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.t()
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
