"""Ablation variants of SIG.

Each subclass changes exactly one thing so the contribution of that component
can be attributed. They live apart from models/sig.py so the model
under study is never edited to run an experiment against it.
"""

import torch as t
import torch.nn.functional as F

from graph_builder import build_permuted_item_graph, build_random_item_graph
from loss_funcs import cal_bpr_loss, cal_infonce_loss, reg_params_squared
from models.sig import SIG
from models.lightgcn_plus_rlmrec import LightGCN_plus_RLMRec


class SIG_PermutedGraph(SIG):
    """PRIMARY semantic control: real graph, node labels shuffled.

    The graph stays isomorphic to the true semantic graph -- same edge count,
    same degree sequence -- so the only thing lost is the correspondence
    between an item and its own semantic neighbourhood. A drop here is
    attributable to semantics and nothing else.

    The InfoNCE anchor still uses the real semantic vectors, so this isolates
    the GRAPH's contribution rather than the LLM embeddings' overall.
    """

    def __init__(self, data_handler, **kwargs):
        super().__init__(data_handler, **kwargs)
        # A local generator prevents graph construction from advancing the
        # global RNG and thereby changing the subsequent DataLoader shuffle.
        control_generator = t.Generator().manual_seed(t.initial_seed() + 104729)
        self.item_graph = build_permuted_item_graph(
            data_handler.item_sem_emb, k=kwargs["knn_k"], device=self.device,
            generator=control_generator,
        )


class SIG_RandomGraph(SIG):
    """Secondary control: uniform random topology.

    Weaker than the permuted control because it also changes the degree
    distribution (measured on yelp: 220k edges and max degree 36, against the
    semantic graph's 170k and 291). Kept as a second data point, not as the
    headline control.
    """

    def __init__(self, data_handler, **kwargs):
        super().__init__(data_handler, **kwargs)
        control_generator = t.Generator().manual_seed(t.initial_seed() + 104729)
        self.item_graph = build_random_item_graph(
            self.item_num, k=kwargs["knn_k"], device=self.device,
            generator=control_generator,
        )


class SIG_NoStructural(SIG):
    """Structural branch removed entirely: h*_i = h_i, no fusion.

    Isolates the whole item-item graph contribution. What remains is the
    collaborative graph plus the item-only InfoNCE anchor.
    """

    def _structural_items(self):
        return t.zeros(
            self.item_num, self.embedding_size, device=self.device
        )


class SIG_NoPrune(SIG):
    """Degree-sensitive pruning disabled (rho forced to 0).

    Isolates Phase 3. The collaborative graph is used in full every epoch.
    """

    def __init__(self, data_handler, **kwargs):
        kwargs = dict(kwargs)
        kwargs["rho"] = 0.0
        super().__init__(data_handler, **kwargs)


class SIG_PosNegAnchor(SIG):
    """Add only RLMRec's negative-item semantic InfoNCE term.

    The graph, fusion, collaborative propagation, pruning, projector, and all
    hyperparameters remain unchanged. This isolates whether positive-only
    anchoring explains the residual Yelp gap.
    """

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self._select_adj())
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(
            user_embeds, item_embeds, batch_data
        )

        bpr_loss = cal_bpr_loss(
            anc_embeds, pos_embeds, neg_embeds
        ) / anc_embeds.shape[0]

        posprf_embeds = self.mlp(self.item_sem_emb[batch_data[1]])
        negprf_embeds = self.mlp(self.item_sem_emb[batch_data[2]])
        kd_loss = (
            cal_infonce_loss(
                pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature
            )
            + cal_infonce_loss(
                neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature
            )
        )
        kd_loss = (kd_loss / anc_embeds.shape[0]) * self.kd_weight

        reg_loss = self.reg_weight * reg_params_squared(self)
        loss = bpr_loss + kd_loss + reg_loss
        return loss, {
            "bpr_loss": bpr_loss,
            "kd_loss": kd_loss,
            "reg_loss": reg_loss,
        }


class LightGCN_plus_RLMRec_Coverage(LightGCN_plus_RLMRec):
    """RLMRec with only a fraction of user profiles available.

    Simulates the operational reality that a deployment never has a fresh
    LLM-generated profile for every user: new arrivals have none yet, and
    existing ones go stale between refreshes. A fraction (1 - profile_coverage)
    of users have their profile replaced by the population mean profile, which
    is the sensible production fallback -- it carries no per-user signal but
    keeps the anchor well-conditioned.

    SIG has no user-profile dependency at all, so its performance is a
    flat line across this sweep by construction. That is the point of the
    comparison.

    Users are selected uniformly at random rather than by activity, so that
    coverage is isolated as a single variable. Ablating the least-active users
    instead would be more realistic but confounds profile availability with
    user difficulty.

    Draws from the global torch RNG, so a seeded run is reproducible and
    different seeds ablate different users.
    """

    def __init__(self, data_handler, profile_coverage=1.0, **kwargs):
        super().__init__(data_handler, **kwargs)
        self.profile_coverage = float(profile_coverage)
        if not 0.0 <= self.profile_coverage <= 1.0:
            raise ValueError("profile_coverage must be in [0, 1]")
        if self.profile_coverage < 1.0:
            n = self.usrprf_embeds.shape[0]
            n_drop = int(round((1.0 - self.profile_coverage) * n))
            if n_drop:
                idx = t.randperm(n, device=self.usrprf_embeds.device)[:n_drop]
                fallback = self.usrprf_embeds.mean(dim=0, keepdim=True)
                self.usrprf_embeds[idx] = fallback
            self.n_profiles_dropped = n_drop
        else:
            self.n_profiles_dropped = 0
