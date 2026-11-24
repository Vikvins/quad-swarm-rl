"""Supervise relative priority and the effect of messages during local conflicts."""

import torch
from torch.nn import functional as F


def conflict_supervision(state, device):
    _, pos, vel, goals, neighbor_ids = state
    pos, vel, goals = (torch.as_tensor(item, device=device) for item in (pos, vel, goals))
    neighbors = torch.as_tensor(neighbor_ids, device=device)
    arena = torch.arange(pos.shape[0], device=device)[:, None, None]

    # Include the intended direction so the auxiliary objective can see
    # crossing paths when all measured velocities are still zero.
    intended = F.normalize(goals - pos, dim=-1)
    predicted_velocity = vel + 0.8 * intended
    delta = pos[arena, neighbors] - pos.unsqueeze(-2)
    delta_v = predicted_velocity[arena, neighbors] - predicted_velocity.unsqueeze(-2)
    closest_time = (-(delta * delta_v).sum(dim=-1) / delta_v.square().sum(dim=-1).clamp_min(1e-6))
    closest = delta + closest_time.clamp(0., 0.65).unsqueeze(-1) * delta_v
    conflicts = (closest.square().sum(dim=-1) < 0.22**2) & (closest_time >= 0.)

    forward = F.normalize(goals[..., :2].mean(dim=1) - pos[..., :2].mean(dim=1), dim=-1)
    progress = (pos[..., :2] * forward[:, None]).sum(dim=-1)
    ahead = progress[arena, neighbors] - progress.unsqueeze(-1)
    drone_ids = torch.arange(pos.shape[1], device=device)[None, :, None]
    neighbor_goes_first = (ahead > .03) | ((ahead.abs() <= .03) & (neighbors < drone_ids))
    return conflicts, neighbor_goes_first


def channel_loss(policy, features, neighbors, enabled, conflicts, neighbor_goes_first):
    hidden = policy.encode(features)
    bids = policy.bid(hidden).squeeze(-1)
    other_bids = torch.gather(bids, -1, neighbors.flatten(-2)).reshape_as(neighbors)
    direction = neighbor_goes_first.to(bids.dtype) * 2 - 1
    edge_mask = conflicts & enabled.unsqueeze(-1).bool()
    rank = (F.softplus(.4 - direction * (other_bids - bids.unsqueeze(-1))) * edge_mask).sum()
    rank /= edge_mask.sum().clamp_min(1)

    with_messages = policy.wait_distribution(hidden, bids, neighbors, enabled).logits
    no_messages = policy.wait_distribution(hidden, torch.zeros_like(bids), neighbors, enabled).logits
    target = (conflicts & neighbor_goes_first).any(dim=-1).to(bids.dtype)
    decision_mask = edge_mask.any(dim=-1)
    active_count = decision_mask.sum().clamp_min(1)
    classification = F.binary_cross_entropy_with_logits(with_messages, target, reduction="none")
    # A message must improve the margin for the appropriate decision over
    # the same hidden state with zeroed messages and the channel flag on.
    improvement = (target * 2 - 1) * (with_messages - no_messages.detach())
    decision = ((classification + F.softplus(.15 - improvement)) * decision_mask).sum() / active_count
    magnitude = (bids.square() * decision_mask).sum() / active_count
    return decision + .5 * rank + .02 * magnitude
