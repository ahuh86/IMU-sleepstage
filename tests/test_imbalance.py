import pytest
import torch

from imu_rnn.imbalance import class_counts, inverse_frequency_weights


def test_training_record_counts_and_inverse_frequency_weights():
    records = [{'label': label} for label in [0] * 8 + [1] * 4 + [2] * 2 + [3]]

    counts = class_counts(records)
    weights = inverse_frequency_weights(counts)

    assert counts == [8, 4, 2, 1]
    torch.testing.assert_close(
        weights,
        torch.tensor([15 / 32, 15 / 16, 15 / 8, 15 / 4], dtype=torch.float32),
    )
    torch.testing.assert_close(weights * torch.tensor(counts), torch.full((4,), 15 / 4))


@pytest.mark.parametrize('counts', ([8, 4, 2, 0], [1, 2, 3], [1, 2, 3, -1]))
def test_inverse_frequency_weights_reject_invalid_counts(counts):
    with pytest.raises(ValueError):
        inverse_frequency_weights(counts)
