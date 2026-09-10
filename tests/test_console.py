import io

from imu_rnn.console import ConsoleProgress, print_metrics, print_epoch
from imu_rnn.metrics import classification_metrics


def test_progress_displays_completed_count_and_actual_metrics():
    stream = io.StringIO()
    with ConsoleProgress('Fold 1/13 Train', 2, stream=stream) as progress:
        progress.update(1, loss=1.25, acc=.5)
        progress.update(2, loss=.75, acc=.625)
    output = stream.getvalue()
    assert 'Fold 1/13 Train' in output
    assert '2/2' in output and '100%' in output
    assert '0.7500' in output and '0.6250' in output


def test_metrics_report_contains_class_names_support_and_undefined_kappa(capsys):
    print_metrics('Test', classification_metrics([0, 0], [0, 0]))
    output = capsys.readouterr().out
    assert 'N/A' in output
    for term in ('Wake', 'Light', 'Deep', 'REM', 'Precision', 'Recall/Sens', 'Specificity',
                 'Balanced-Acc', 'F1', 'Support', 'Confusion'):
        assert term in output


def test_epoch_report_displays_training_validation_and_best(capsys):
    train = dict(loss=.5, accuracy=.75, seconds=3.)
    val = classification_metrics([0,1], [0,1])
    print_epoch(2, 30, train, val, .6, 2, 0, 5, True)
    output = capsys.readouterr().out
    for term in ('Epoch 2/30', 'Train Loss=0.5000', 'Val Loss=0.6000',
                 'Macro-F1', 'Kappa', 'Best=2', '0/5', 'saved'):
        assert term in output
