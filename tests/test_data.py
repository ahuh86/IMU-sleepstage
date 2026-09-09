from pathlib import Path

import numpy as np
import pytest
import torch

from imu_rnn.data import (build_index, load_window, split_subjects, fit_normalizer,
                          WindowDataset, collate_sequences, evenly_spaced_indices,
                          prepare_window_store)


def window(root, subject='a', number=0, start=0., n=3750, value=1., labels=None):
    path = root / subject / '10s-step' / f'npy_{number}.npy'
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((n, 8), dtype=np.float64)
    arr[:, 0] = start + np.arange(n) * .008
    arr[:, 1:7] = value
    arr[:, 7] = 0 if labels is None else labels
    np.save(path, arr)
    return path


@pytest.mark.parametrize('n', [3749, 3750, 3751])
def test_length_adjustment(tmp_path, n):
    path = window(tmp_path, n=n)
    arr = np.load(path)
    arr[-1, 1:7] = 9
    np.save(path, arr)
    x = load_window(path)
    assert x.shape == (6, 3750)
    assert x[0, -1] == (1 if n == 3751 else 9)
    if n == 3749:
        assert torch.equal(x[:, -1], x[:, -2])


@pytest.mark.parametrize('kind', ['length', 'nan_axis', 'label_fraction', 'label_range', 'nan_label', 'time_gap'])
def test_reject_invalid_window(tmp_path, kind):
    path = window(tmp_path, n=3748 if kind == 'length' else 3750)
    arr = np.load(path)
    if kind == 'nan_axis': arr[0, 1] = np.nan
    if kind == 'label_fraction': arr[0, 7] = .5
    if kind == 'label_range': arr[0, 7] = 4
    if kind == 'nan_label': arr[0, 7] = np.nan
    if kind == 'time_gap': arr[20:, 0] += .02
    np.save(path, arr)
    with pytest.raises(ValueError):
        load_window(path)


def test_index_ties_numeric_order_gaps_and_hash_refresh(tmp_path):
    labels = np.r_[np.full(1875, 3), np.full(1875, 1)]
    p = window(tmp_path, number=8, labels=labels)
    window(tmp_path, number=9, start=10)
    window(tmp_path, number=11, start=20)
    window(tmp_path, number=12, start=50)
    records = build_index(tmp_path, tmp_path/'cache')['a']
    assert [r['id'] for r in records] == ['npy_8', 'npy_9', 'npy_11', 'npy_12']
    assert [r['segment'] for r in records] == [0, 0, 1, 2]
    assert records[0]['label'] == 3
    arr = np.load(p); arr[:, 1] = 5; np.save(p, arr)
    updated = build_index(tmp_path, tmp_path/'cache')['a'][0]
    assert updated['sha256'] != records[0]['sha256']
    assert updated['sum'][0] == 18750


def test_split_and_train_only_statistics(tmp_path):
    for subject, value in [('a', 1), ('b', 3), ('c', 100), ('d', 1000)]:
        window(tmp_path, subject=subject, value=value)
    index = build_index(tmp_path, tmp_path/'cache')
    assert split_subjects(index, 'c') == {'train': ['a', 'b'], 'val': ['d'], 'test': ['c']}
    assert split_subjects(index, 'd')['val'] == ['a']
    norm = fit_normalizer(index['a'] + index['b'])
    assert norm['mean'] == [2.] * 6
    assert norm['std'] == [1.] * 6


def test_causal_sequences_boundaries_and_collation(tmp_path):
    for i in range(5): window(tmp_path, number=i, start=i*10, value=i)
    window(tmp_path, number=6, start=60, value=6)
    window(tmp_path, subject='b', value=77)
    index = build_index(tmp_path, tmp_path/'cache')
    records = index['a'] + index['b']
    ds = WindowDataset(records, {'mean': [0.]*6, 'std': [1.]*6}, seq_len=3)
    assert ds[0][0].shape[0] == 1
    assert ds[4][0][:, 0, 0].tolist() == [2., 3., 4.]
    assert ds[5][0][:, 0, 0].tolist() == [6.]
    assert ds[6][0][:, 0, 0].tolist() == [77.]
    batch, lengths, labels, meta = collate_sequences([ds[0], ds[4]])
    assert batch.shape == (2, 3, 6, 3750)
    assert lengths.tolist() == [1, 3]
    assert labels.tolist() == [0, 0]
    assert [r['id'] for r in meta] == ['npy_0', 'npy_4']
    selected = evenly_spaced_indices(records, 3)
    assert selected == [0, 2, 5, 6]
    subset = WindowDataset(records, {'mean': [0.]*6, 'std': [1.]*6}, target_indices=[4])
    assert len(subset) == 1
    assert subset[0][0][:, 0, 0].tolist() == [0., 1., 2., 3., 4.]
    assert subset.context_indices(0) == [0, 1, 2, 3, 4]
    assert ds.context_indices(4) == [2, 3, 4]


def test_packed_store_equivalence_reuse_and_source_invalidation(tmp_path):
    path = window(tmp_path, n=3749, value=3)
    window(tmp_path, number=1, start=10, n=3751, value=5)
    cache = tmp_path/'cache'
    index = build_index(tmp_path, cache)
    normalizer = {'mean': [1.]*6, 'std': [2.]*6}
    expected = WindowDataset(index['a'], normalizer)[1][0]
    packed_path = prepare_window_store(index, cache)
    packed = np.load(packed_path, mmap_mode='r')
    assert packed.shape == (2, 6, 3750)
    assert packed.dtype == np.float32
    dataset = WindowDataset(index['a'], normalizer)
    assert dataset.normalizer == normalizer
    assert torch.equal(dataset[1][0], expected)
    # Pack remains usable even without source files; it contains immutable raw axes.
    moved = path.with_suffix('.original')
    path.rename(moved)
    assert torch.equal(WindowDataset(index['a'], normalizer)[1][0], expected)
    moved.rename(path)
    assert prepare_window_store(index, cache) == packed_path
    arr = np.load(path); arr[:, 1] = 7; np.save(path, arr)
    refreshed = build_index(tmp_path, cache)
    new_path = prepare_window_store(refreshed, cache)
    assert new_path != packed_path
    assert WindowDataset(refreshed['a'], normalizer)[0][0][0, 0, 0] == 3
