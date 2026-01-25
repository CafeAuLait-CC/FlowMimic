from itertools import cycle


def balanced_batch_iter(loader_a, loader_b, ratio_a=1, ratio_b=1):
    if ratio_a <= 0 or ratio_b <= 0:
        raise ValueError("ratio_a and ratio_b must be positive")

    iter_a = cycle(loader_a)
    iter_b = cycle(loader_b)

    while True:
        batch_parts = []
        for _ in range(ratio_a):
            batch_parts.append(next(iter_a))
        for _ in range(ratio_b):
            batch_parts.append(next(iter_b))
        yield batch_parts
