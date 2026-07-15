import numpy as np


def sample_condition_frame_count(
    seq_len,
    cond_frames_min=None,
    cond_frames_max=None,
    choices=None,
    choice_probs=None,
):
    """Sample a valid condition count and return it with the padding budget."""
    seq_len = max(1, int(seq_len))
    if choices:
        counts = [max(1, min(int(value), seq_len)) for value in choices]
        if len(set(counts)) != len(counts):
            raise ValueError(f"Condition frame choices must be unique: {counts}")
        weights = None
        if choice_probs is not None:
            weights = [float(value) for value in choice_probs]
            if len(weights) != len(counts):
                raise ValueError(
                    "Condition frame choice probabilities must match the choices"
                )
            if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
                raise ValueError(
                    "Condition frame choice probabilities must be non-negative "
                    "and have a positive sum"
                )
        return random_choice(counts, weights), max(counts)

    min_frames = max(1, min(int(cond_frames_min or 1), seq_len))
    max_frames = max(
        min_frames,
        min(int(cond_frames_max or min_frames), seq_len),
    )
    if max_frames > min_frames:
        return int(np.random.randint(min_frames, max_frames + 1)), max_frames
    return min_frames, max_frames


def random_choice(values, weights=None):
    if weights is None:
        return int(values[np.random.randint(0, len(values))])
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(values, p=probabilities))


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
