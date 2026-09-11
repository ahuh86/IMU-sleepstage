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


def test_square_root_weakens_inverse_frequency_ratios():
    full = inverse_frequency_weights([8, 4, 2, 1])
    weakened = inverse_frequency_weights([8, 4, 2, 1], power=.5)

    torch.testing.assert_close(weakened, full.sqrt())
    assert weakened[-1] / weakened[0] < full[-1] / full[0]


@pytest.mark.parametrize(
    ('counts', 'power'),
    [([8, 4, 2, 0], 1.), ([1, 2, 3], 1.), ([1, 2, 3, -1], 1.),
     ([8, 4, 2, 1], 0.), ([8, 4, 2, 1], 1.1)],
)
def test_inverse_frequency_weights_reject_invalid_inputs(counts, power):
    with pytest.raises(ValueError):
        inverse_frequency_weights(counts, power=power)
