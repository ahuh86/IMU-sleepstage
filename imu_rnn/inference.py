"""Checkpoint-only inference on validated split-data-format windows."""
from pathlib import Path
import torch

from .data import build_index, WindowDataset
from .engine import evaluate_dataset, seed_everything
from .experiment import resolve_device
from .io import write_predictions
from .model import CNNRNN


def run_inference(checkpoint_path, data_root, output_csv, device='auto', subjects=None, cache_dir=None):
    device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    seed_everything(checkpoint['config']['seed'])
    torch.set_num_threads(checkpoint['config'].get('threads', 2))
    model = CNNRNN(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    index = build_index(Path(data_root), Path(cache_dir or Path(output_csv).parent/'inference_cache'))
    names = subjects or sorted(index)
    if len(names) != len(set(names)) or any(name not in index for name in names):
        raise ValueError('Requested subjects must be unique and present in the input')
    rows = []
    for name in names:
        dataset = WindowDataset(index[name], checkpoint['normalizer'], checkpoint['config']['seq_len'])
        predictions, _ = evaluate_dataset(model, dataset, device)
        rows.extend(predictions)
    write_predictions(output_csv, rows)
    return rows
