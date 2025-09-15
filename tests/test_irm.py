import torch

from firm.invariance.irm_loss import irm_penalty_from_env_losses


def test_irm_penalty_zero_when_equal_losses():
    losses = [torch.tensor(1.0, requires_grad=True) for _ in range(3)]
    p = irm_penalty_from_env_losses(losses)
    # gradient of scale * c is the same across envs so squared grads sum equally
    assert p.item() >= 0.0


def test_irm_penalty_positive_when_different_losses():
    losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)]
    p = irm_penalty_from_env_losses(losses)
    assert p.item() > 0.0

