"""Human-readable live console output, following the local DeepSleepNet runner."""
import sys

from tqdm import tqdm

from .metrics import CLASS_NAMES


class ConsoleProgress:
    def __init__(self, label, total, stream=None, unit='batch'):
        self.bar = tqdm(total=total, desc=label, unit=unit, dynamic_ncols=True,
                        mininterval=.3, ascii=True, file=stream or sys.stdout)

    def update(self, completed, **metrics):
        self.bar.set_postfix({k: f'{v:.4f}' for k, v in metrics.items()}, refresh=False)
        self.bar.update(completed-self.bar.n)

    def close(self):
        self.bar.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def format_score(value):
    return 'N/A' if value is None else f'{value:.4f}'


def print_epoch(epoch, maximum, train, validation, val_loss, best_epoch, stale, patience, improved):
    print(f'Epoch {epoch}/{maximum} | Train Loss={train["loss"]:.4f} Acc={train["accuracy"]:.4f} | '
          f'Val Loss={val_loss:.4f} Acc={validation["accuracy"]:.4f} '
          f'Macro-F1={validation["macro_f1"]:.4f} Kappa={format_score(validation["kappa"])} | '
          f'Best={best_epoch} Patience={stale}/{patience}'
          + (' | best checkpoint saved' if improved else ''), flush=True)


def print_metrics(title, metrics):
    print(f'\n{title}\n' + '='*60, flush=True)
    print(f'Accuracy={metrics["accuracy"]:.4f}  Balanced-Acc={metrics["balanced_accuracy"]:.4f}  '
          f'Macro-F1={metrics["macro_f1"]:.4f}  '
          f'Kappa={format_score(metrics["kappa"])}  Windows={metrics["n"]}')
    print(f'{"Class":<10}{"Precision":>12}{"Recall/Sens":>14}{"Specificity":>14}{"F1":>12}{"Support":>10}')
    for i, name in enumerate(CLASS_NAMES):
        print(f'{name:<10}{metrics["precision"][i]:>12.4f}{metrics["recall"][i]:>14.4f}'
              f'{metrics["specificity"][i]:>14.4f}{metrics["f1"][i]:>12.4f}'
              f'{metrics["support"][i]:>10}')
    print('Confusion matrix (rows=true, columns=predicted):')
    print(f'{"":10}' + ''.join(f'{name:>10}' for name in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, metrics['confusion_matrix']):
        print(f'{name:<10}' + ''.join(f'{value:>10}' for value in row))
    print(flush=True)


def evaluate_with_progress(model, dataset, device, label):
    from .engine import evaluate_dataset
    bar, phase = None, None
    diagnostics = {}

    def update(event):
        nonlocal bar, phase
        if event['phase'] != phase:
            if bar is not None:
                bar.close()
            phase = event['phase']
            bar = ConsoleProgress(f'{label} {phase}', event['total'], unit='window')
        values = {name: event[key] for name, key in [('loss','loss'), ('acc','accuracy')] if key in event}
        bar.update(event['completed'], **values)
    try:
        rows, metrics = evaluate_dataset(model, dataset, device, progress=update, diagnostics=diagnostics)
        return rows, metrics, diagnostics
    finally:
        if bar is not None:
            bar.close()
