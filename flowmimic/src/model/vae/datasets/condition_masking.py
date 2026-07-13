import numpy as np


def make_condition_frame_drop_mask(
    valid_len,
    prob,
    mode="random",
    max_block_frac=0.25,
):
    """Return a boolean mask of condition frames to drop.

    The mask always keeps at least one valid frame when valid_len > 1.
    """
    valid_len = int(valid_len)
    prob = float(prob or 0.0)
    if valid_len <= 1 or prob <= 0.0:
        return np.zeros((max(valid_len, 0),), dtype=bool)

    prob = min(max(prob, 0.0), 1.0)
    target = int(round(valid_len * prob))
    target = max(1, min(target, valid_len - 1))
    mode = (mode or "random").lower()
    if mode == "mixed":
        mode = "block" if np.random.rand() < 0.5 else "random"

    if mode == "random":
        frame_drop = np.zeros(valid_len, dtype=bool)
        frame_drop[np.random.choice(valid_len, size=target, replace=False)] = True
        return frame_drop

    if mode != "block":
        raise ValueError(f"Unsupported condition frame drop mode: {mode}")

    frame_drop = np.zeros(valid_len, dtype=bool)
    max_block_frac = min(max(float(max_block_frac or 0.25), 0.01), 1.0)
    max_block = max(1, min(target, int(round(valid_len * max_block_frac))))
    remaining = target
    attempts = 0
    while remaining > 0 and attempts < valid_len * 8:
        block_len = np.random.randint(1, min(max_block, remaining) + 1)
        start = np.random.randint(0, valid_len - block_len + 1)
        block = slice(start, start + block_len)
        newly_dropped = int((~frame_drop[block]).sum())
        if newly_dropped > 0:
            frame_drop[block] = True
            remaining -= newly_dropped
        attempts += 1

    if remaining > 0:
        candidates = np.flatnonzero(~frame_drop)
        add = np.random.choice(candidates, size=min(remaining, candidates.size), replace=False)
        frame_drop[add] = True

    if frame_drop.all():
        frame_drop[np.random.randint(0, valid_len)] = False
    return frame_drop
