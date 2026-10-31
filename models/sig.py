import torch as t
from torch import nn
import torch.nn.functional as F

from graph_builder import build_frozen_item_graph
from loss_funcs import cal_bpr_loss, cal_infonce_loss, reg_params_squared
from utils import degree_sensitive_prune, spmm

init = nn.init.xavier_uniform_


class SIG(nn.Module):
    """SIG: LLM semantic anchoring over a frozen item-item graph.

    Two propagations run in parallel: item ID embeddings over a frozen semantic
    kNN graph (structural), and user+item ID embeddings over a degree-pruned
    collaborative graph (collaborative). They are fused additively, and an
    InfoNCE term anchors the fused item embedding to the projected LLM vector.
    """

    def __init__(self, data_handler, embedding_size, layer_num, ii_layer_num,
                 knn_k, rho, reg_weight, kd_weight, kd_temperature):
        super(SIG, self).__init__()
        self.user_num = data_handler.user_num
        self.item_num = data_handler.item_num
        self.embedding_size = embedding_size
        self.layer_num = layer_num          # L_ui
        self.ii_layer_num = ii_layer_num    # L_ii
        self.rho = rho
        self.reg_weight = reg_weight        # beta
        self.kd_weight = kd_weight          # alpha
        self.kd_temperature = kd_temperature  # tau

        device = data_handler.device
        self.device = device
        self.trn_mat = data_handler.trn_mat
        self.adj = data_handler.torch_adj   # full graph, used at evaluation

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, embedding_size)))

        if data_handler.item_sem_emb is None:
            raise ValueError(
                "SIG requires item semantic embeddings; "
                "itm_emb_np.pkl was not found for this dataset."
            )
        # Frozen: registered as a buffer, never a Parameter, so no gradient and
        # no appearance in reg_params_squared.
        self.register_buffer(
            "item_sem_emb",
            t.as_tensor(data_handler.item_sem_emb).float().to(device),
            persistent=False,
        )
        self.register_buffer(
            "item_graph",
            build_frozen_item_graph(data_handler.item_sem_emb, k=knn_k, device=device),
            persistent=False,
        )

        sem_dim = self.item_sem_emb.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(sem_dim, (sem_dim + embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((sem_dim + embedding_size) // 2, embedding_size),
        )

        # Seeded from PyTorch's global RNG state (itself seeded by main.py's
        # init_seed) rather than a hardcoded constant, so torch.manual_seed(s)
        # propagates and a multi-seed ablation doesn't silently reuse the same
        # prune sequence across seeds. Kept as an explicit Generator so
        # pruning stays reproducible within a single run.
        self.generator = t.Generator().manual_seed(
            int(t.randint(0, 2**63 - 1, (1,)).item())
        )
        self.train_adj = None
        self.final_embeds = None
        self.is_training = False

        self._init_weight()
        self.on_epoch_start()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def reset_cache(self):
        """Drop the cached propagation.

        final_embeds is not part of state_dict, so it must be cleared whenever
        the parameters change, or evaluation serves embeddings belonging to a
        different set of weights.
        """
        self.final_embeds = None
        self.is_training = False

    def on_epoch_start(self):
        """Resample the degree-pruned collaborative graph (Phase 3)."""
        self.train_adj = degree_sensitive_prune(
            self.trn_mat, self.rho, self.device, self.generator
        )
        self.reset_cache()

    def _select_adj(self):
        """Training sees the pruned graph; evaluation sees the full one."""
        return self.train_adj if self.is_training else self.adj

    def _structural_items(self):
        """Phase 4.1: propagate item IDs over the frozen semantic graph.

        Eq. 6 takes the FINAL layer, not a mean over layers.
        """
        h = self.item_embeds
        for _ in range(self.ii_layer_num):
            h = spmm(self.item_graph, h)
        return F.normalize(h, p=2, dim=1)

    def _collaborative(self, adj):
        """Phase 4.2: propagate user+item IDs over the collaborative graph.

        Eqs. 9-10 mean-pool layers 0..L_ui.
        """
        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        for _ in range(self.layer_num):
            embeds_list.append(spmm(adj, embeds_list[-1]))
        return sum(embeds_list) / len(embeds_list)

    def _collaborative_only_items(self):
        """Item embeddings before structural fusion. Used by tests."""
        return self._collaborative(self._select_adj())[self.user_num:]

    def forward(self, adj=None, keep_rate=1.0):
        """Phases 4.2-4.3. keep_rate is accepted for interface parity and unused;
        denoising is handled by on_epoch_start()."""
        if not self.is_training and self.final_embeds is not None:
            return (
                self.final_embeds[: self.user_num],
                self.final_embeds[self.user_num:],
            )

        adj = self._select_adj() if adj is None else adj
        collab = self._collaborative(adj)
        user_embeds = collab[: self.user_num]
        item_embeds = collab[self.user_num:] + self._structural_items()

        embeds = t.concat([user_embeds, item_embeds], dim=0)
        if not self.is_training:
            # Only cache evaluation-time embeddings. The training pass runs on
            # the pruned graph and must never be reused for prediction.
            self.final_embeds = embeds
        return user_embeds, item_embeds

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        return user_embeds[ancs], item_embeds[poss], item_embeds[negs]

    def cal_loss(self, batch_data):
        """Phase 5: L = BPR + alpha * InfoNCE + beta * ||Theta||_2^2."""
        self.is_training = True
        user_embeds, item_embeds = self.forward(self._select_adj())
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(
            user_embeds, item_embeds, batch_data
        )

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]

        # Anchor positive items only, on the FUSED embedding, with in-batch
        # negatives -- Eq. 14.
        posprf_embeds = self.mlp(self.item_sem_emb[batch_data[1]])
        kd_loss = cal_infonce_loss(
            pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature
        )
        kd_loss = (kd_loss / anc_embeds.shape[0]) * self.kd_weight

        reg_loss = self.reg_weight * reg_params_squared(self)

        loss = bpr_loss + kd_loss + reg_loss
        return loss, {"bpr_loss": bpr_loss, "kd_loss": kd_loss, "reg_loss": reg_loss}

    def _mask_predict(self, full_preds, train_mask):
        return full_preds * (1 - train_mask) - 1e8 * train_mask

    def full_predict(self, batch_data):
        # is_training must be cleared BEFORE forward, or the first evaluation
        # batch takes the training path.
        self.is_training = False
        user_embeds, item_embeds = self.forward()

        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        full_preds = user_embeds[pck_users] @ item_embeds.T
        return self._mask_predict(full_preds, train_mask)
