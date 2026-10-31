import torch as t
import torch.nn.functional as F


def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return t.sum(F.softplus(neg_preds - pos_preds))


def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / t.sqrt(
        1e-8 + all_embeds2.square().sum(-1, keepdim=True)
    )
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = t.log(
        t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1)
    )
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss


def reg_params(model):
    """
    Calculate L2 regularization loss for the model parameters.

    Args:
        model (torch.nn.Module): The model containing the parameters to regularize.

    Returns:
        torch.Tensor: The regularization loss.
    """
    reg_loss = t.sum(t.stack([W.norm(2) for W in model.parameters()]))
    return reg_loss


# def reg_params(model):
#     reg_loss = 0
#     for W in model.parameters():
#         reg_loss += W.norm(2).square()
#     return reg_loss

def reg_params_squared(model):
    """Sum of squared L2 norms over all parameters.

    This is the ||Theta||_2^2 term of the SIG objective. It is kept
    separate from reg_params(), which sums un-squared norms, so that existing
    baselines keep producing the numbers they always produced.
    """
    return t.sum(t.stack([W.norm(2).square() for W in model.parameters()]))
