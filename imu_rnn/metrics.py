"""Fixed four-class metrics with explicit undefined Cohen kappa."""
import numpy as np

CLASS_NAMES = ['Wake', 'Light', 'Deep', 'REM']


def classification_metrics(truth, prediction):
    y, p = np.asarray(truth), np.asarray(prediction)
    if y.ndim != 1 or p.shape != y.shape or not y.size:
        raise ValueError('truth and prediction must be equally sized nonempty vectors')
    for values in (y, p):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or np.any((values < 0) | (values > 3)):
            raise ValueError('labels must be integers in 0..3')
    cm = np.bincount(4 * y.astype(int) + p.astype(int), minlength=16).reshape(4, 4)
    support, predicted = cm.sum(1), cm.sum(0)
    tp = cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(4), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(4), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(4), where=(support + predicted) != 0)
    accuracy = float(tp.sum() / y.size)
    chance = float(np.dot(support.astype(float), predicted) / y.size ** 2)
    kappa = float((accuracy - chance) / (1 - chance)) if chance < 1 else None
    return dict(accuracy=accuracy, macro_f1=float(f1.mean()), kappa=kappa,
                precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(),
                support=support.tolist(), confusion_matrix=cm.tolist(), n=int(y.size))
