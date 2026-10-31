import torch as t
from torch import nn
import torchvision.models as models
from loss_funcs import cal_bpr_loss, reg_params, cal_infonce_loss
from utils import SpAdjEdgeDrop, spmm

# initialization functions
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
        self.embedding_size = embedding_size  # Embedding size
        self.keep_rate = keep_rate  # Dropout keep rate for edges
        self.layer_num = layer_num  # Number of GCN layers
        self.reg_weight = reg_weight  # Regularization weight
        self.kd_weight = kd_weight  # Knowledge distillation weight
        self.kd_temperature = kd_temperature  # Temperature for distillation loss
        self.user_sem_emb = (
            data_handler.user_sem_emb
        )  # Text semantic embeddings for users
        self.item_sem_emb = (
            data_handler.item_sem_emb
        )  # Text semantic embeddings for items

        # Load adjacency matrix and initialize embeddings
        self.adj = data_handler.torch_adj
        self.user_embeds = nn.Parameter(
            init(t.empty(self.user_num, self.embedding_size))
        )
        self.item_embeds = nn.Parameter(
            init(t.empty(self.item_num, self.embedding_size))
        )

        # Edge dropper for training with dropout
        self.edge_dropper = SpAdjEdgeDrop()

        # Semantic-embeddings (text-based)
        device = data_handler.device
        self.usrprf_embeds = t.tensor(self.user_sem_emb).float().to(device)
        self.itmprf_embeds = t.tensor(self.item_sem_emb).float().to(device)

        # MLP for transforming the profile (text) embeddings
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

        # ----------------------------------------------------------
        # Image branch: using a pre-trained ResNet18 as the encoder
        self.img_encoder = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        # Remove the final classification layer so that we get a feature vector.
        self.img_encoder.fc = nn.Identity()
        # Freeze the encoder. reg_params() sums over every parameter, so an
        # unfrozen ResNet would be pulled towards zero by the regularisation
        # term alone -- even though the image branch is inactive whenever
        # cal_loss is called without item_images.
        self.img_encoder.requires_grad_(False)
        # Projection layer to map ResNet18 output (512-dim) to embedding_size.
        self.img_projection = nn.Linear(512, self.embedding_size)
        # ----------------------------------------------------------

        self.final_embeds = None
        self.is_training = False

        self._init_weight()

    def _init_weight(self):
        """Initialize weights for the MLP layers."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        # Optionally, initialize the image projection layer.
        init(self.img_projection.weight)

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
        """Forward pass for LightGCN_plus.

        Note: The image branch is not fused into the propagation here.
        Image embeddings are used only for contrastive alignment in the loss.
        """
        if adj is None:
            adj = self.adj

        if not self.is_training and self.final_embeds is not None:
            return (
                self.final_embeds[: self.user_num],
                self.final_embeds[self.user_num :],
            )

        # Concatenate user and item embeddings for propagation.
        embeds = t.concat(
            [self.user_embeds, self.item_embeds], dim=0
        )  # Concatenation along the first dimension
        embeds_list = [embeds]

        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)

        # Propagate embeddings across layers.
        for _ in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        # Average the embeddings from all layers.
        embeds = sum(embeds_list) / len(embeds_list)
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

    def get_item_image_embedding(self, images):
        """
        Compute image embeddings for a batch of images.
        Expects images of shape (batch_size, C, H, W).
        """
        # Extract raw image features via the CNN encoder.
        raw_features = self.img_encoder(images)  # e.g., output shape: (batch_size, 512)
        # Project the raw features to the common embedding space.
        return self.img_projection(raw_features)

    def cal_loss(self, batch_data, item_images=None):
        """
        Calculate the BPR loss, regularization loss, and knowledge distillation loss.
        Optionally, if item_images is provided (a tensor of shape [num_items, C, H, W]),
        compute an additional contrastive loss between item CF embeddings and image embeddings.
        """
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(
            user_embeds, item_embeds, batch_data
        )

        # Transform the profile (text) embeddings using the MLP.
        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(
            usrprf_embeds, itmprf_embeds, batch_data
        )

        # Calculate BPR loss.
        bpr_loss = (
            cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        )
        reg_loss = self.reg_weight * reg_params(self)

        # Calculate knowledge distillation (contrastive) loss for text.
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
        kd_loss = (kd_loss / anc_embeds.shape[0]) * self.kd_weight

        # ----------------------------------------------------------
        # Compute image-based contrastive loss if item_images are provided.
        # Assumes item_images is a tensor of shape [num_items, C, H, W],
        # so that item_images[poss] gives images for positive items and similarly for negatives.
        kd_img_loss = 0.0
        if item_images is not None:
            # Compute image embeddings for the positive and negative items.
            pos_img_embeds = self.get_item_image_embedding(
                item_images[pos_embeds.new_tensor(batch_data[1], dtype=t.long)]
            )
            neg_img_embeds = self.get_item_image_embedding(
                item_images[neg_embeds.new_tensor(batch_data[2], dtype=t.long)]
            )
            # Compute contrastive loss between CF item embeddings and image embeddings.
            kd_img_loss = cal_infonce_loss(
                pos_embeds, pos_img_embeds, pos_img_embeds, self.kd_temperature
            ) + cal_infonce_loss(
                neg_embeds, neg_img_embeds, neg_img_embeds, self.kd_temperature
            )
            kd_img_loss = (kd_img_loss / anc_embeds.shape[0]) * self.kd_weight
        # ----------------------------------------------------------

        # Total loss.
        total_loss = bpr_loss + reg_loss + kd_loss + kd_img_loss
        losses = {
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "kd_loss": kd_loss,
            "kd_img_loss": kd_img_loss,
        }
        return total_loss, losses

    def _mask_predict(self, full_preds, train_mask):
        """Apply a mask to predictions, hiding training data."""
        return full_preds * (1 - train_mask) - 1e8 * train_mask

    def full_predict(self, batch_data):
        """Full prediction for evaluation."""
        self.is_training = False
        user_embeds, item_embeds = self.forward(self.adj, 1.0)

        # Unpack the batch data.
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        # Calculate predictions and mask out training pairs.
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)

        return full_preds
