"""Atomic artifacts; predictions are sufficient to reconstruct all metrics."""
import csv
import json
import os
from pathlib import Path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def write_predictions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    fields = ['subject', 'id', 'start', 'end', 'label', 'prediction', 'p0', 'p1', 'p2', 'p3']
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_predictions(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ('label', 'prediction'):
            row[key] = int(row[key]) if row[key] else None
        for key in ('start', 'end', 'p0', 'p1', 'p2', 'p3'):
            row[key] = float(row[key])
    return rows
