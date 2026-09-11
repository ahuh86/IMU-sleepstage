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


def test_training_reports_every_batch_with_sample_weighted_metrics():
    torch.set_num_threads(2)
    seed_everything(42)
    model = CNNRNN()
    batches = [batch(), batch()]
    # Unequal final batch catches an unweighted average of batch accuracies/losses.
    batches[1] = tuple(v[:1] for v in batches[1])
    updates = []
    result = train_epoch(model, batches, torch.optim.Adam(model.parameters()),
                         torch.device('cpu'), progress=updates.append)
    assert [u['step'] for u in updates] == [1, 2]
    assert updates[-1]['samples'] == 3
    assert updates[-1]['accuracy'] == result['accuracy']
    assert updates[-1]['loss'] == result['loss']
    assert updates[-1]['loss'] == pytest.approx(
        (updates[0]['batch_loss'] * 2 + updates[1]['batch_loss']) / 3)
    assert updates[-1]['lr'] == .001


def test_training_uses_supplied_weighted_criterion():
    class ConstantLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([2., 1., 0., -1.]))

        def forward(self, x, lengths):
            return self.logits.expand(len(x), -1)

    model = ConstantLogits()
    labels = torch.tensor([0, 1])
    training_batch = (torch.zeros(2, 1), torch.ones(2, dtype=torch.long), labels, [{}, {}])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 8., 1., 1.]))
    expected = float(criterion(model.logits.expand(2, -1), labels))

    result = train_epoch(
        model,
        [training_batch],
        torch.optim.SGD(model.parameters(), lr=0.),
        torch.device('cpu'),
        criterion=criterion,
    )

    assert result['loss'] == pytest.approx(expected)


def test_weighted_epoch_loss_uses_sum_of_target_weights_across_batches():
    class ConstantLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([2., 1., 0., -1.]))

        def forward(self, x, lengths):
            return self.logits.expand(len(x), -1)

    model = ConstantLogits()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 8., 1., 1.]))
    labels = torch.tensor([0, 0, 1])
    batches = [
        (torch.zeros(2, 1), torch.ones(2, dtype=torch.long), labels[:2], [{}, {}]),
        (torch.zeros(1, 1), torch.ones(1, dtype=torch.long), labels[2:], [{}]),
    ]
    expected = float(criterion(model.logits.expand(3, -1), labels))

    result = train_epoch(
        model, batches, torch.optim.SGD(model.parameters(), lr=0.),
        torch.device('cpu'), criterion=criterion,
    )

    assert result['loss'] == pytest.approx(expected)
