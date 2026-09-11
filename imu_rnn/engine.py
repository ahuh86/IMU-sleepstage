import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from .metrics import classification_metrics


def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@dataclass
class BestTracker:
    patience: int = 5
    best: float = -1.
    best_epoch: int = 0
    stale: int = 0

    def update(self, score, epoch):
        if not np.isfinite(score):
            raise ValueError('Validation score is not finite')
        if score > self.best:
            self.best, self.best_epoch, self.stale = float(score), epoch, 0
            return True
        self.stale += 1
        return False

    @property
    def should_stop(self):
        return self.stale >= self.patience


def train_epoch(model, loader, optimizer, device, progress=None, criterion=None):
    model.train()
    criterion = criterion or torch.nn.CrossEntropyLoss()
    total_loss, loss_denominator, count, correct = 0., 0., 0, 0
    started = time.monotonic()
    for step, (x, lengths, y, _) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        n = len(y)
        criterion_weight = getattr(criterion, 'weight', None)
        batch_denominator = (float(criterion_weight[y].sum())
                             if criterion_weight is not None else n)
        total_loss += float(loss.detach()) * batch_denominator
        loss_denominator += batch_denominator
        correct += int((logits.argmax(1) == y).sum())
        count += n
        if progress:
            progress(dict(step=step, total_steps=len(loader), samples=count,
                          loss=total_loss/loss_denominator, batch_loss=float(loss.detach()),
                          accuracy=correct/count, lr=optimizer.param_groups[0]['lr'],
                          elapsed_seconds=time.monotonic()-started))
    if not count:
        raise ValueError('Empty training loader')
    return dict(loss=total_loss/loss_denominator, accuracy=correct/count, n=count,
                seconds=time.monotonic()-started)


def prediction_rows(probabilities, labels, metadata):
    rows = []
    for probability, label, meta in zip(probabilities, labels, metadata):
        row = {key: meta[key] for key in ('subject', 'id', 'start', 'end')}
        row.update(label=int(label), prediction=int(np.argmax(probability)))
        row.update({f'p{i}': float(probability[i]) for i in range(4)})
        rows.append(row)
    return rows


def metrics_for_rows(rows):
    keys = [(row['subject'], row['id']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate prediction targets')
    return classification_metrics([r['label'] for r in rows], [r['prediction'] for r in rows])


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for x, lengths, y, metadata in loader:
        probabilities = model(x.to(device), lengths).softmax(1).cpu().numpy()
        rows.extend(prediction_rows(probabilities, y.tolist(), metadata))
    return rows, metrics_for_rows(rows)


@torch.inference_mode()
def evaluate_dataset(model, dataset, device, batch_size=128, progress=None, diagnostics=None):
    """Encode each needed window once with frozen BN; reuse only within this call.

    Equivalent to ordinary eval, never used for gradient training. The feature
    cache is rebuilt after every epoch, so it cannot contain stale CNN weights.
    """
    from .data import load_window
    model.eval()
    started = time.monotonic()
    contexts = [dataset.context_indices(i) for i in range(len(dataset))]
    needed = sorted({i for context in contexts for i in context})
    feature_cache = {}
    mean = torch.tensor(dataset.normalizer['mean'], dtype=torch.float32)[:, None]
    std = torch.tensor(dataset.normalizer['std'], dtype=torch.float32)[:, None]
    for start in range(0, len(needed), batch_size):
        indices = needed[start:start+batch_size]
        windows = torch.stack([(load_window(dataset.records[i]['path'])-mean)/std for i in indices])
        encoded = model.encode_windows(windows.to(device)).cpu()
        feature_cache.update(zip(indices, encoded))
        if progress:
            progress(dict(phase='encode', completed=start+len(indices), total=len(needed)))
    rows = []
    total_loss, correct = 0., 0
    for start in range(0, len(contexts), batch_size):
        chunk = contexts[start:start+batch_size]
        sequences = [torch.stack([feature_cache[i] for i in context]) for context in chunk]
        lengths = torch.tensor([len(s) for s in sequences])
        features = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).to(device)
        logits = model.classify_features(features, lengths)
        probabilities = logits.softmax(1).cpu().numpy()
        metadata = [dataset.records[c[-1]] for c in chunk]
        labels = torch.tensor([r['label'] for r in metadata], device=device)
        total_loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum'))
        correct += int((logits.argmax(1) == labels).sum())
        rows.extend(prediction_rows(probabilities, [r['label'] for r in metadata], metadata))
        if progress:
            progress(dict(phase='predict', completed=len(rows), total=len(dataset),
                          loss=total_loss/len(rows), accuracy=correct/len(rows)))
    if len(rows) != len(dataset):
        raise RuntimeError('Prediction count differs from target count')
    metrics = metrics_for_rows(rows)
    if diagnostics is not None:
        diagnostics.update(loss=total_loss/len(rows), seconds=time.monotonic()-started)
    return rows, metrics
