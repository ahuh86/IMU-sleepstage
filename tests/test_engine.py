import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from imu_rnn.engine import BestTracker, seed_everything, train_epoch, evaluate
from imu_rnn.io import save_json, write_predictions, read_predictions
from imu_rnn.metrics import classification_metrics
from imu_rnn.model import CNNRNN


def batch():
    return (torch.randn(2, 2, 6, 128), torch.tensor([1, 2]), torch.tensor([0, 1]),
            [dict(subject='s', id=f'npy_{i}', start=i*10., end=i*10+30., label=i) for i in range(2)])


def test_best_tracker_keeps_earliest_tie_and_stops_after_five():
    tracker = BestTracker(patience=5)
    assert tracker.update(.2, 1)
    assert not tracker.update(.2, 2)
    assert tracker.update(.3, 3)
    for epoch in range(4, 8):
        assert not tracker.update(.1, epoch)
        assert not tracker.should_stop
    tracker.update(.3, 8)
    assert tracker.should_stop
    assert tracker.best_epoch == 3


def test_real_training_changes_weights_and_evaluation_csv_reproduces_metrics(tmp_path):
    torch.set_num_threads(2)
    seed_everything(42)
    model = CNNRNN()
    before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    b = batch()
    result = train_epoch(model, [b], optimizer, torch.device('cpu'))
    assert np.isfinite(result['loss'])
    assert not torch.equal(before, model.classifier.weight)
    rows, metrics = evaluate(model, [b], torch.device('cpu'))
    assert len(rows) == 2
    assert all(sum(row[f'p{i}'] for i in range(4)) == pytest.approx(1.) for row in rows)
    write_predictions(tmp_path / 'pred.csv', rows)
    restored = read_predictions(tmp_path / 'pred.csv')
    assert classification_metrics([r['label'] for r in restored], [r['prediction'] for r in restored]) == metrics
    save_json(tmp_path / 'metrics.json', metrics)


def test_evaluation_rejects_repeated_targets():
    b = batch()
    with pytest.raises(ValueError, match='Duplicate'):
        evaluate(CNNRNN(), [b, b], torch.device('cpu'))
