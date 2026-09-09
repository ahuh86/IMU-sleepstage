import copy
import numpy as np
import pytest
import torch

from imu_rnn.model import CNNRNN
from imu_rnn.metrics import classification_metrics


def test_padding_never_enters_cnn_or_rnn_and_backward_is_finite():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = CNNRNN()
    other = copy.deepcopy(model)
    x = torch.randn(2, 3, 6, 128)
    lengths = torch.tensor([1, 3])
    changed = x.clone()
    changed[0, 1:] = float('nan')
    actual = model(x, lengths)
    expected = other(changed, lengths)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 4)
    actual.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for a, b in zip(model.buffers(), other.buffers()):
        torch.testing.assert_close(a, b)


def test_checkpoint_roundtrip_and_short_sequence_batch_invariance(tmp_path):
    model = CNNRNN().eval()
    x = torch.randn(2, 3, 6, 128)
    lengths = torch.tensor([1, 3])
    with torch.no_grad():
        expected = model(x, lengths)
        short = model(x[:1, :1], torch.tensor([1]))
    torch.testing.assert_close(expected[:1], short, atol=1e-6, rtol=1e-5)
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    restored = CNNRNN().eval()
    restored.load_state_dict(torch.load(tmp_path / 'model.pt', weights_only=True))
    torch.testing.assert_close(restored(x, lengths), expected)


@pytest.mark.parametrize('lengths', [torch.tensor([0, 2]), torch.tensor([4, 2])])
def test_invalid_lengths_rejected(lengths):
    with pytest.raises(ValueError):
        CNNRNN()(torch.zeros(2, 3, 6, 128), lengths)


def test_metrics_fixed_four_classes_and_missing_class():
    m = classification_metrics([0, 0, 1, 2], [0, 1, 1, 2])
    assert m['accuracy'] == .75
    assert m['macro_f1'] == pytest.approx(7 / 12)
    assert m['kappa'] == pytest.approx(7 / 11)
    assert m['support'] == [2, 1, 1, 0]
    assert m['recall'] == [.5, 1., 1., 0.]
    assert np.array(m['confusion_matrix']).shape == (4, 4)


def test_metrics_degenerate_kappa_is_explicitly_unavailable():
    assert classification_metrics([0, 0], [0, 0])['kappa'] is None


def test_metrics_reject_invalid_or_empty_inputs():
    for truth, pred in [([], []), ([4], [0]), ([0], [0, 1]), ([.5], [0])]:
        with pytest.raises(ValueError):
            classification_metrics(truth, pred)
