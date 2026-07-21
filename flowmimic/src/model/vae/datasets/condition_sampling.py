import hashlib
import os

import numpy as np


CONDITION_PATTERNS = ("even", "random", "boundary_gap")


def condition_sample_key(path, camera=None, start=0, seq_len=196):
    name = os.path.splitext(os.path.basename(str(path)))[0]
    camera = "none" if camera is None else str(camera)
    return f"{name}|camera={camera}|start={int(start)}|len={int(seq_len)}"


def stable_condition_seed(base_seed, sample_key):
    payload = f"{int(base_seed)}|{sample_key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % (2**32)


def sample_condition_indices(seq_len, count, pattern="even", rng=None):
    seq_len = max(1, int(seq_len))
    count = max(1, min(int(count), seq_len))
    pattern = str(pattern).strip().lower().replace("-", "_")
    if pattern not in CONDITION_PATTERNS:
        raise ValueError(
            f"Unsupported condition pattern {pattern!r}; "
            f"expected one of {CONDITION_PATTERNS}"
        )
    if count == seq_len:
        return np.arange(seq_len, dtype=np.int64)

    rng = rng if rng is not None else np.random
    if pattern == "even":
        indices = np.linspace(0, seq_len - 1, count)
        indices = np.round(indices).astype(np.int64)
    elif pattern == "random":
        indices = np.sort(rng.choice(seq_len, size=count, replace=False)).astype(
            np.int64
        )
    else:
        indices = _sample_boundary_gap_indices(seq_len, count, rng)

    indices = np.unique(indices)
    if indices.size != count:
        raise RuntimeError(
            f"Condition pattern {pattern!r} produced {indices.size} indices, "
            f"expected {count}"
        )
    return indices


def deterministic_condition_indices(
    seq_len,
    count,
    pattern,
    base_seed,
    sample_key,
):
    seed = stable_condition_seed(base_seed, sample_key)
    return sample_condition_indices(
        seq_len,
        count,
        pattern=pattern,
        rng=np.random.RandomState(seed),
    )


def _sample_boundary_gap_indices(seq_len, count, rng):
    if count == 1:
        return np.asarray([0], dtype=np.int64)

    # Preserve clip boundaries while reserving a contiguous interior interval
    # with no observations. The gap spans 25-50% of the sequence when the
    # requested condition count leaves enough unobserved frames.
    available_gap = seq_len - count
    if available_gap <= 0:
        return np.arange(seq_len, dtype=np.int64)
    preferred_min = max(1, int(round(0.25 * seq_len)))
    preferred_max = max(preferred_min, int(round(0.50 * seq_len)))
    gap_max = min(available_gap, preferred_max, max(1, seq_len - 2))
    gap_min = min(preferred_min, gap_max)
    if gap_max > gap_min:
        gap_len = int(rng.randint(gap_min, gap_max + 1))
    else:
        gap_len = gap_min

    start_max = seq_len - gap_len - 1
    gap_start = int(rng.randint(1, start_max + 1)) if start_max >= 1 else 1
    gap_end = gap_start + gap_len
    candidates = np.concatenate(
        [
            np.arange(1, gap_start, dtype=np.int64),
            np.arange(gap_end, seq_len - 1, dtype=np.int64),
        ]
    )
    needed = count - 2
    if candidates.size < needed:
        raise RuntimeError(
            f"Boundary-gap pattern has {candidates.size} candidates for "
            f"{needed} requested interior anchors"
        )
    interior = (
        np.sort(rng.choice(candidates, size=needed, replace=False))
        if needed > 0
        else np.empty((0,), dtype=np.int64)
    )
    return np.concatenate(
        [np.asarray([0], dtype=np.int64), interior, np.asarray([seq_len - 1])]
    )
