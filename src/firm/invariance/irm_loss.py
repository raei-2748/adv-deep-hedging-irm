from __future__ import annotations

from typing import Iterable

import torch


def irm_penalty_from_env_losses(env_losses: Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Compute a simple IRM-like penalty given per-environment scalar losses.

    Uses the IRMv1 trick of a dummy scalar 'scale' parameter and computes the
    squared gradients across environments; then sums them.
    """
    scale = torch.tensor(1.0, requires_grad=True, device=next(iter(env_losses)).device)
    penalties = []
    for loss in env_losses:
        # risk(w * f(x), y) is approximated by 'scale * loss' here (surrogate)
        g = torch.autograd.grad((scale * loss), [scale], create_graph=True)[0]
        penalties.append(g.pow(2))
    if not penalties:
        return torch.tensor(0.0)
    return torch.stack(penalties).sum()

