import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

from imu_rnn.data import build_index, WindowDataset, collate_sequences, fit_normalizer
from imu_rnn.engine import evaluate, evaluate_dataset
from imu_rnn.model import CNNRNN
from imu_rnn.experiment import Config, run_experiment
from imu_rnn.inference import run_inference


def fixture_data(root):
    for subject in ('a', 'b', 'c'):
        folder = root / subject / '10s-step'
        folder.mkdir(parents=True)
        for i in range(3):
            a = np.zeros((3750, 8))
            a[:, 0] = np.arange(3750)*.008 + i*10
            a[:, 1:7] = np.sin(np.arange(3750)[:, None]*.02) + ord(subject)-97
            a[:, 7] = i
            np.save(folder / f'npy_{i+1}.npy', a)


def test_cached_eval_matches_window_eval(tmp_path):
    torch.set_num_threads(2)
    fixture_data(tmp_path/'data')
    index = build_index(tmp_path/'data', tmp_path/'cache')
    records = index['a']
    dataset = WindowDataset(records, fit_normalizer(index['b']), seq_len=3)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_sequences)
    model = CNNRNN().eval()
    rows, metrics = evaluate(model, loader, torch.device('cpu'))
    fast_rows, fast_metrics = evaluate_dataset(model, dataset, torch.device('cpu'), batch_size=2)
    assert metrics == fast_metrics
    np.testing.assert_allclose([[r[f'p{i}'] for i in range(4)] for r in rows],
                               [[r[f'p{i}'] for i in range(4)] for r in fast_rows], atol=1e-6)


def test_full_synthetic_loso_artifacts_resume_and_independent_inference(tmp_path):
    torch.set_num_threads(2)
    fixture_data(tmp_path/'data')
    config = Config(data_root=str(tmp_path/'data'), output_dir=str(tmp_path/'run'),
                    cache_dir=str(tmp_path/'cache'), epochs=1, batch_size=2, seq_len=2,
                    device='cpu', all_subjects=True)
    summary = run_experiment(config)
    assert summary['complete']
    assert len(summary['folds']) == 3
    assert summary['pooled']['n'] == 9
    for fold in summary['folds']:
        assert fold['metrics']['n'] == 3
        assert len(fold['split']['train']) == 1
        assert set(fold['split']['train']).isdisjoint(fold['split']['test'] + fold['split']['val'])
    checkpoint = tmp_path/'run'/'a'/'best.pt'
    stamp = checkpoint.stat().st_mtime_ns
    config.resume = True
    assert run_experiment(config) == summary
    assert checkpoint.stat().st_mtime_ns == stamp
    rows = run_inference(checkpoint, tmp_path/'data', tmp_path/'infer.csv', 'cpu', subjects=['a'], cache_dir=tmp_path/'cache')
    assert len(rows) == 3
    assert [r['id'] for r in rows] == ['npy_1','npy_2','npy_3']
    cli_output = tmp_path/'cli_predictions.csv'
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'predict.py'),
                    '--checkpoint', str(checkpoint), '--data-root', str(tmp_path/'data'),
                    '--output-csv', str(cli_output), '--subjects', 'a', '--device', 'cpu',
                    '--cache-dir', str(tmp_path/'cache')], check=True, capture_output=True, text=True)
    assert cli_output.read_text() == (tmp_path/'infer.csv').read_text()
    config.seq_len = 3
    with pytest.raises(ValueError, match='configuration|fingerprint'):
        run_experiment(config)


def test_invalid_config_does_not_start_run(tmp_path):
    with pytest.raises(ValueError):
        run_experiment(Config(data_root=str(tmp_path), output_dir=str(tmp_path/'out'), epochs=0))
    assert not (tmp_path/'out').exists()
