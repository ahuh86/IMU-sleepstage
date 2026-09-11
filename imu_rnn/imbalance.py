"""Training-only class distribution and loss weighting."""

import torch


def class_counts(records, num_classes=4):
    counts = [0] * num_classes
    for record in records:
        label = record.get('label')
        if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < num_classes:
            raise ValueError(f'Invalid class label in training records: {label!r}')
        counts[label] += 1
    return counts


def inverse_frequency_weights(counts, num_classes=4):
    """Return N / (C * n_c), whose sample-weighted mean is one."""
    if len(counts) != num_classes or any(
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
            for count in counts):
        raise ValueError(f'All {num_classes} classes need a positive integer training count')
    total = sum(counts)
    return torch.tensor(
        [total / (num_classes * count) for count in counts],
        dtype=torch.float32,
    )
