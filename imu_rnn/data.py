"""Validated, causal IMU window loading without subject or temporal leakage."""
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path
import re
import uuid

import numpy as np
import torch
from torch.utils.data import Dataset


def _read(path):
    path = Path(path)
    try:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] < 8 or len(arr) not in (3749, 3750, 3751):
            raise ValueError('expected 3749/3750/3751 rows and at least 8 columns')
        arr = np.asarray(arr[:, :8], dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError('nonfinite time, IMU value or label')
        if not np.allclose(np.diff(arr[:, 0]), .008, atol=1e-5, rtol=0):
            raise ValueError('non-contiguous 125 Hz timestamps')
        labels = arr[:, 7]
        if not ((labels == np.floor(labels)) & (labels >= 0) & (labels <= 3)).all():
            raise ValueError('sleep labels must be integers in 0..3')
        return arr
    except (ValueError, TypeError, OSError) as exc:
        raise ValueError(f'{path}: {exc}') from exc


def load_window(path):
    arr = _read(path)
    axes = arr[:3750, 1:7]
    if len(axes) == 3749:
        axes = np.concatenate([axes, axes[-1:]], axis=0)
    return torch.from_numpy(np.ascontiguousarray(axes.T, dtype=np.float32))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_index(data_root: Path, cache_dir: Path):
    root, cache = Path(data_root).resolve(), Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / 'window_index.json'
    previous = {}
    if cache_file.exists():
        try:
            saved = json.loads(cache_file.read_text(encoding='utf-8'))
            if saved.get('version') == 1:
                previous = {r['path']: r for rows in saved['index'].values() for r in rows}
        except (ValueError, KeyError, TypeError):
            previous = {}
    index, total = {}, 0
    for folder in sorted(root.iterdir()):
        source = folder / '10s-step'
        if not source.is_dir():
            continue
        paths = list(source.glob('npy_*.npy'))
        if not paths:
            continue
        def number(path):
            match = re.fullmatch(r'npy_(\d+)', path.stem)
            if match is None:
                raise ValueError(f'Invalid window filename: {path}')
            return int(match.group(1))
        paths.sort(key=number)
        rows, segment, last_number, last_start = [], 0, None, None
        for path in paths:
            current_number = number(path)
            if last_number == current_number:
                raise ValueError(f'Duplicate numeric window index: {path}')
            absolute, digest = str(path.resolve()), _sha256(path)
            cached = previous.get(absolute)
            if cached and cached['sha256'] == digest:
                record = dict(cached)
            else:
                arr = _read(path)
                axes = arr[:, 1:7]
                counts = Counter(arr[:, 7].astype(int).tolist())
                record = dict(path=absolute, id=path.stem, start=float(arr[0, 0]),
                              end=float(arr[-1, 0]), label=int(counts.most_common(1)[0][0]),
                              sha256=digest, count=len(arr), sum=axes.sum(axis=0).tolist(),
                              sumsq=np.square(axes).sum(axis=0).tolist())
            if last_number is not None and (current_number != last_number + 1 or
                                            abs(record['start'] - last_start - 10) > .02):
                segment += 1
            record.update(subject=folder.name, segment=segment)
            rows.append(record)
            last_number, last_start = current_number, record['start']
            total += 1
            if total % 1000 == 0:
                print(f'Validated and hashed {total} windows', flush=True)
        index[folder.name] = rows
    if not index:
        raise ValueError(f'No subject/10s-step/npy_*.npy files found under {root}')
    temp = cache_file.with_suffix('.tmp')
    temp.write_text(json.dumps({'version': 1, 'index': index}), encoding='utf-8')
    temp.replace(cache_file)
    return index


def split_subjects(index, test_subject):
    subjects = sorted(index)
    if len(subjects) < 3:
        raise ValueError('LOSO train/validation/test needs at least three subjects')
    if test_subject not in subjects:
        raise ValueError(f'Unknown test subject: {test_subject}')
    val = subjects[(subjects.index(test_subject) + 1) % len(subjects)]
    return {'train': [s for s in subjects if s not in (test_subject, val)],
            'val': [val], 'test': [test_subject]}


def fit_normalizer(records):
    if not records:
        raise ValueError('Cannot fit normalization on an empty training set')
    count = sum(r['count'] for r in records)
    mean = np.sum([r['sum'] for r in records], axis=0) / count
    variance = np.maximum(np.sum([r['sumsq'] for r in records], axis=0) / count - mean**2, 0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.
    return {'mean': mean.tolist(), 'std': std.tolist()}


def prepare_window_store(index, cache_dir):
    """Pack immutable raw axes into a content-addressed, disk-backed array."""
    records = [record for subject in sorted(index) for record in index[subject]]
    if not records:
        raise ValueError('Cannot pack an empty index')
    identity = json.dumps({'version': 1,
                           'sources': [(r['path'], r['sha256']) for r in records]},
                          ensure_ascii=True, separators=(',', ':')).encode('utf-8')
    key = hashlib.sha256(identity).hexdigest()
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f'windows_{key}.npy'
    shape = (len(records), 6, 3750)
    valid = False
    if destination.exists():
        try:
            existing = np.load(destination, mmap_mode='r', allow_pickle=False)
            valid = existing.shape == shape and existing.dtype == np.float32
            del existing
        except (ValueError, OSError):
            pass
    if not valid:
        temporary = cache / f'windows_{key}.{uuid.uuid4().hex}.tmp'
        array = None
        try:
            array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=shape)
            for position, record in enumerate(records):
                array[position] = load_window(record['path']).numpy()
                if (position + 1) % 1000 == 0:
                    print(f'Packed {position + 1}/{len(records)} windows', flush=True)
            array.flush()
            del array
            array = None
            temporary.replace(destination)
        finally:
            if array is not None:
                del array
            if temporary.exists():
                temporary.unlink()
    for position, record in enumerate(records):
        record.update(store_index=position, store_path=str(destination))
    return destination


class WindowDataset(Dataset):
    def __init__(self, records, normalizer, seq_len=21, target_indices=None):
        if seq_len < 1:
            raise ValueError('seq_len must be positive')
        self.records = records
        self.normalizer = {key: list(normalizer[key]) for key in ('mean', 'std')}
        self.seq_len = seq_len
        self.target_indices = list(range(len(records))) if target_indices is None else list(target_indices)
        if len(set(self.target_indices)) != len(self.target_indices) or any(i < 0 or i >= len(records) for i in self.target_indices):
            raise ValueError('Target indices must be unique and in range')
        self.mean = torch.tensor(normalizer['mean'], dtype=torch.float32)[:, None]
        self.std = torch.tensor(normalizer['std'], dtype=torch.float32)[:, None]
        if self.mean.shape != (6, 1) or self.std.shape != (6, 1) or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Normalizer must contain six finite means and positive standard deviations')
        self._cache = OrderedDict()
        self._stores = OrderedDict()
        self._history = []
        groups = {}
        for i, record in enumerate(records):
            key = (record['subject'], record['segment'])
            history = groups.setdefault(key, [])
            if history and record['start'] <= records[history[-1]]['start']:
                raise ValueError('Records within each subject segment must be chronological')
            history.append(i)
            self._history.append(tuple(history[-seq_len:]))

    def __len__(self):
        return len(self.target_indices)

    def context_indices(self, index):
        """Return record indices for the dataset item (after target subsetting)."""
        return list(self._history[self.target_indices[index]])

    def __getitem__(self, item):
        target = self.target_indices[item]
        frames = []
        for index in self._history[target]:
            source = self.records[index]
            if 'store_path' in source and 'store_index' in source:
                store_path = source['store_path']
                if store_path not in self._stores:
                    self._stores[store_path] = np.load(store_path, mmap_mode='r', allow_pickle=False)
                    if len(self._stores) > 4:
                        self._stores.popitem(last=False)
                self._stores.move_to_end(store_path)
                raw = torch.from_numpy(self._stores[store_path][source['store_index']].copy())
                frames.append((raw - self.mean) / self.std)
                continue
            path = source['path']
            if path not in self._cache:
                self._cache[path] = load_window(path)
                if len(self._cache) > 256:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(path)
            frames.append((self._cache[path] - self.mean) / self.std)
        record = self.records[target]
        return torch.stack(frames), int(record['label']), record


def collate_sequences(batch):
    sequences, labels, metadata = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    x = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return x, lengths, torch.tensor(labels, dtype=torch.long), list(metadata)


def evenly_spaced_indices(records, per_subject):
    if per_subject < 1:
        raise ValueError('per_subject must be positive')
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(record['subject'], []).append(i)
    chosen = []
    for rows in groups.values():
        positions = np.linspace(0, len(rows)-1, min(per_subject, len(rows)), dtype=int)
        chosen.extend(rows[i] for i in positions)
    return chosen
