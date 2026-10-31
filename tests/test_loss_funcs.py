import torch
from torch import nn

from loss_funcs import reg_params, reg_params_squared


def test_squared_norm_is_sum_of_squared_l2_norms():
    m = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]))
    # single parameter, frobenius norm 5 -> squared 25
    assert torch.isclose(reg_params_squared(m), torch.tensor(25.0))


def test_differs_from_unsquared_version():
    m = nn.Linear(4, 4, bias=False)
    assert not torch.isclose(reg_params_squared(m), reg_params(m))


def test_is_differentiable():
    m = nn.Linear(3, 3, bias=False)
    reg_params_squared(m).backward()
    assert m.weight.grad is not None
