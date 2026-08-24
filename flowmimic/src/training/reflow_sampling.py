"""Sampling helpers for endpoint-aware guided reflow training."""

from __future__ import annotations

import torch


FLOW_TIME_CATEGORY_NAMES = (
    "uniform",
    "exact_start",
    "near_start",
    "near_end",
    "exact_end",
)


def normalized_flow_time_probabilities(config: dict) -> tuple[float, ...]:
    probabilities = tuple(
        float(config.get(name, 0.0)) for name in FLOW_TIME_CATEGORY_NAMES
    )
    if any(value < 0.0 for value in probabilities):
        raise ValueError("Flow-time probabilities cannot be negative")
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("Flow-time probabilities must have a positive sum")
    near_width = float(config.get("near_width", 0.1))
    if not 0.0 < near_width <= 0.5:
        raise ValueError("Flow-time near_width must lie in (0, 0.5]")
    return tuple(value / total for value in probabilities)


def sample_reflow_times(
    batch_size: int,
    config: dict,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample flow times and return their categorical source IDs."""
    probabilities = normalized_flow_time_probabilities(config)
    probability_tensor = torch.tensor(
        probabilities,
        dtype=torch.float32,
        device=device,
    )
    categories = torch.multinomial(
        probability_tensor,
        int(batch_size),
        replacement=True,
    )
    random_values = torch.rand(int(batch_size), dtype=dtype, device=device)
    times = random_values.clone()
    near_width = float(config.get("near_width", 0.1))
    exact_start = categories == FLOW_TIME_CATEGORY_NAMES.index("exact_start")
    near_start = categories == FLOW_TIME_CATEGORY_NAMES.index("near_start")
    near_end = categories == FLOW_TIME_CATEGORY_NAMES.index("near_end")
    exact_end = categories == FLOW_TIME_CATEGORY_NAMES.index("exact_end")
    times[exact_start] = 0.0
    times[near_start] = random_values[near_start] * near_width
    times[near_end] = 1.0 - random_values[near_end] * near_width
    times[exact_end] = 1.0
    return times, categories
