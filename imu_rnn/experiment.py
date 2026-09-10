"""Subject-isolated experiments with resumable completed folds."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (build_index, split_subjects, fit_normalizer, WindowDataset,
                   collate_sequences, evenly_spaced_indices, prepare_window_store)
from .engine import seed_everything, BestTracker, train_epoch, evaluate_dataset, metrics_for_rows
from .io import save_json, write_predictions, read_predictions
from .model import CNNRNN
from .console import ConsoleProgress, evaluate_with_progress, print_epoch, print_metrics, format_score


@dataclass
class Config:
    data_root: str = str(Path(__file__).resolve().parents[2] / 'split-data')
    output_dir: str = 'results/loso'
    cache_dir: str = 'results/cache'
    test_subject: str | None = None
    all_subjects: bool = False
    smoke: bool = False
    epochs: int = 30
    batch_size: int = 8
    lr: float = .001
    seq_len: int = 21
    patience: int = 5
    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 0
    threads: int = 2
    resume: bool = False

    def validate(self):
        for field in ('epochs', 'batch_size', 'seq_len', 'patience', 'threads'):
            if getattr(self, field) < 1:
                raise ValueError(f'{field} must be positive')
        if not np.isfinite(self.lr) or self.lr <= 0 or self.num_workers < 0:
            raise ValueError('Invalid learning rate or worker count')
        if self.all_subjects and self.test_subject is not None:
            raise ValueError('Choose either all subjects or one test subject')
        if not self.all_subjects and not self.test_subject and not self.smoke:
            raise ValueError('Specify --all-subjects or --test-subject (or --smoke)')


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu, cuda or auto')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(name)


def source_hashes():
    root = Path(__file__).resolve().parent
    paths = sorted(root.glob('*.py')) + [root.parent/'train_loso.py', root.parent/'predict.py']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def environment(device):
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                deterministic_algorithms=True, cpu_threads=torch.get_num_threads())


def run_fold(config, index, subject, output, device, fingerprint, data_digest):
    seed_everything(config.seed)
    fold_dir = output / subject
    fold_dir.mkdir(parents=True, exist_ok=True)
    split = split_subjects(index, subject)
    records = {part: [r for s in names for r in index[s]] for part, names in split.items()}
    normalizer = fit_normalizer(records['train'])
    datasets = {}
    for part, rows in records.items():
        chosen = evenly_spaced_indices(rows, 8 if part == 'train' else 32) if config.smoke else None
        datasets[part] = WindowDataset(rows, normalizer, config.seq_len, chosen)
    print(f'Train subjects: {", ".join(split["train"])}\n'
          f'Validation: {split["val"][0]} | Test: {subject}\n'
          f'Target windows: Train={len(datasets["train"])} Val={len(datasets["val"])} '
          f'Test={len(datasets["test"])}', flush=True)
    save_json(fold_dir/'split.json', split)
    save_json(fold_dir/'normalizer.json', normalizer)
    loader = DataLoader(datasets['train'], batch_size=config.batch_size, shuffle=True,
                        num_workers=config.num_workers, collate_fn=collate_sequences,
                        generator=torch.Generator().manual_seed(config.seed))
    model = CNNRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    tracker, history = BestTracker(config.patience), []
    started = time.monotonic()
    maximum = 1 if config.smoke else config.epochs
    for epoch in range(1, maximum+1):
        with ConsoleProgress(f'{subject} Train Epoch {epoch}/{maximum}', len(loader)) as bar:
            def progress(details):
                bar.update(details['step'], batch_loss=details['batch_loss'],
                           avg_loss=details['loss'], acc=details['accuracy'], lr=details['lr'])
                # Console updates per batch, disk status stays inexpensive.
                if details['step'] == 1 or details['step'] % 100 == 0 or details['step'] == details['total_steps']:
                    save_json(output/'status.json', dict(state='training', subject=subject, epoch=epoch,
                              updated=datetime.now(timezone.utc).isoformat(), **details))
            train = train_epoch(model, loader, optimizer, device, progress)
        save_json(output/'status.json', dict(state='validating', subject=subject, epoch=epoch))
        _, val, val_diagnostics = evaluate_with_progress(model, datasets['val'], device, 'Validation')
        val['loss'] = val_diagnostics['loss']
        improved = tracker.update(val['macro_f1'], epoch)
        if improved:
            checkpoint = dict(model_state=model.state_dict(), normalizer=normalizer,
                              config=asdict(config), model_config={'feature_dim': 32, 'hidden_dim': 32},
                              split=split, epoch=epoch, validation_metrics=val,
                              fingerprint=fingerprint, data_digest=data_digest,
                              environment=environment(device))
            torch.save(checkpoint, fold_dir/'best.pt.tmp')
            os.replace(fold_dir/'best.pt.tmp', fold_dir/'best.pt')
        history.append(dict(epoch=epoch, train=train, validation=val, best=improved))
        save_json(fold_dir/'history.json', history)
        print_epoch(epoch, maximum, train, val, val['loss'], tracker.best_epoch,
                    tracker.stale, tracker.patience, improved)
        if tracker.should_stop:
            print(f'Early stopping: best epoch={tracker.best_epoch}, '
                  f'validation Macro-F1={tracker.best:.4f}', flush=True)
            break
    checkpoint = torch.load(fold_dir/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    save_json(output/'status.json', dict(state='testing_best_checkpoint', subject=subject,
                                       best_epoch=tracker.best_epoch))
    # This is the ONLY use of test predictions in a fold.
    print(f'Restored best epoch {tracker.best_epoch}; evaluating test subject {subject}', flush=True)
    rows, metrics, _ = evaluate_with_progress(model, datasets['test'], device, 'Test')
    expected = {(r['subject'], r['id']) for r in records['test']}
    if not config.smoke and {(r['subject'], r['id']) for r in rows} != expected:
        raise RuntimeError('Test predictions do not cover every unique target')
    write_predictions(fold_dir/'predictions.csv', rows)
    result = dict(test_subject=subject, split=split, metrics=metrics,
                  best_epoch=tracker.best_epoch, best_validation_macro_f1=tracker.best,
                  epochs_run=len(history), seconds=time.monotonic()-started,
                  target_counts={k: len(v) for k, v in datasets.items()},
                  fingerprint=fingerprint, complete=True, smoke=config.smoke)
    save_json(fold_dir/'result.json', result)
    print_metrics(f'Completed test subject: {subject}', metrics)
    return result


def summarize(folds, output, subjects, smoke):
    rows = [row for fold in folds for row in read_predictions(output/fold['test_subject']/'predictions.csv')]
    between = {}
    for metric in ('accuracy', 'macro_f1', 'kappa'):
        values = [f['metrics'][metric] for f in folds if f['metrics'][metric] is not None]
        between[metric] = dict(mean=float(np.mean(values)) if values else None,
                               std=float(np.std(values)) if values else None, n=len(values))
    summary = dict(complete=len(folds) == len(subjects), smoke=smoke,
                   planned_subjects=subjects, folds=folds, between_folds=between,
                   pooled=metrics_for_rows(rows),
                   protocol='30-second overlapping windows with 10-second stride; each target counted once',
                   caveat='Label semantics and original PSG synchronization remain unverified')
    save_json(output/'summary.json', summary)
    lines = ['# CNN + unidirectional RNN results', '',
             f'Completed folds: {len(folds)}/{len(subjects)}. Smoke test: {smoke}.', '',
             summary['protocol']+'.', '', summary['caveat']+'.', '',
             '| Test subject | Accuracy | Macro-F1 | Kappa | Best epoch | Windows |',
             '|---|---:|---:|---:|---:|---:|']
    for fold in folds:
        m = fold['metrics']
        kappa = f'{m["kappa"]:.5f}' if m['kappa'] is not None else 'undefined'
        lines.append(f'| {fold["test_subject"]} | {m["accuracy"]:.5f} | {m["macro_f1"]:.5f} | '
                     f'{kappa} | {fold["best_epoch"]} | {m["n"]} |')
    lines += ['', 'Fold mean/std (population standard deviation; undefined kappa excluded):',
              '```json', json.dumps(between, indent=2), '```', '',
              'Pooled metrics, per-class precision/recall/F1/support and confusion matrix:',
              '```json', json.dumps(summary['pooled'], indent=2), '```', '',
              'Class order: Wake, Light, Deep, REM. Absent-class F1 is 0; support is reported.',
              'Per-fold per-class metrics and matrices are in each result.json and summary.json.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def run_experiment(config):
    config.validate()
    device = resolve_device(config.device)
    torch.set_num_threads(config.threads)
    seed_everything(config.seed)
    output = Path(config.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not config.resume:
        raise ValueError('Output directory is nonempty; use a new directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    print(f'Device: {device} | Batch size: {config.batch_size} | Sequence length: {config.seq_len}\n'
          f'Max epochs: {1 if config.smoke else config.epochs} | LR: {config.lr} | Seed: {config.seed}\n'
          f'Data: {Path(config.data_root).resolve()}\nOutput: {output}\n'
          'Preparing data: validating hashes and window cache...', flush=True)
    save_json(output/'status.json', dict(state='indexing', started=datetime.now(timezone.utc).isoformat()))
    try:
        index = build_index(Path(config.data_root), Path(config.cache_dir))
        subjects = sorted(index) if config.all_subjects else [config.test_subject or sorted(index)[0]]
        for subject in subjects:
            split_subjects(index, subject)
        manifest = [{k: r[k] for k in ('subject', 'id', 'path', 'start', 'end', 'label', 'segment', 'sha256')}
                    for records in index.values() for r in records]
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        identity = asdict(config)
        for key in ('resume', 'output_dir', 'cache_dir'):
            identity.pop(key)
        identity['data_root'] = str(Path(config.data_root).resolve())
        metadata = dict(config=identity, data_digest=digest, source_hashes=source_hashes())
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        if (output/'run.json').exists():
            old = json.loads((output/'run.json').read_text(encoding='utf-8'))
            if old['fingerprint'] != fingerprint:
                raise ValueError('Run fingerprint mismatch: configuration, source code or data changed')
        else:
            save_json(output/'run.json', dict(**metadata, fingerprint=fingerprint, environment=environment(device),
                      started=datetime.now(timezone.utc).isoformat(), smoke=config.smoke))
            save_json(output/'manifest.json', manifest)
        prepare_window_store(index, Path(config.cache_dir))
        folds = []
        for fold_number, subject in enumerate(subjects, 1):
            print(f'\n{"="*60}\nFold {fold_number}/{len(subjects)} | Test subject: {subject}\n'
                  f'{"="*60}', flush=True)
            result_file = output/subject/'result.json'
            if config.resume and result_file.exists():
                result = json.loads(result_file.read_text(encoding='utf-8'))
                if not result.get('complete') or result['fingerprint'] != fingerprint:
                    raise ValueError('Completed fold fingerprint mismatch')
                saved_rows = read_predictions(output/subject/'predictions.csv')
                if metrics_for_rows(saved_rows) != result['metrics']:
                    raise ValueError('Saved prediction metrics do not match completed fold')
                if not (output/subject/'best.pt').exists():
                    raise ValueError('Completed fold has no checkpoint')
                print(f'Reusing completed fold {subject}', flush=True)
            else:
                result = run_fold(config, index, subject, output, device, fingerprint, digest)
            folds.append(result)
            summary = summarize(folds, output, subjects, config.smoke)
        save_json(output/'status.json', dict(state='complete', folds=len(folds), smoke=config.smoke,
                                           finished=datetime.now(timezone.utc).isoformat()))
        print(f'\nCompleted {len(folds)} folds. Fold mean +/- std:', flush=True)
        for name, stats in summary['between_folds'].items():
            print(f'{name}: {format_score(stats["mean"])} +/- {format_score(stats["std"])} (n={stats["n"]})')
        print_metrics('Pooled evaluation across completed folds', summary['pooled'])
        return summary
    except BaseException as exc:
        save_json(output/'status.json', dict(state='failed', error=f'{type(exc).__name__}: {exc}',
                                           updated=datetime.now(timezone.utc).isoformat()))
        raise
