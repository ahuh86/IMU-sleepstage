"""Shared per-window CNN followed by a single, forward-only vanilla RNN."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class CNNRNN(nn.Module):
    def __init__(self, feature_dim=32, hidden_dim=32):
        super().__init__()
        self.window_encoder = nn.Sequential(
            nn.Conv1d(6, 64, 25, stride=2, padding=12),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 15, stride=2, padding=7),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(128, feature_dim), nn.ReLU(),
        )
        self.rnn = nn.RNN(feature_dim, hidden_dim, num_layers=1,
                          nonlinearity='tanh', batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, 4)

    def encode_windows(self, windows):
        return self.window_encoder(windows)

    def classify_features(self, features, lengths):
        packed = pack_padded_sequence(features, lengths.detach().cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        return self.classifier(hidden[-1])

    def forward(self, x, lengths):
        if x.ndim != 4 or x.shape[2] != 6:
            raise ValueError('x must have shape [batch, sequence, 6, time]')
        if lengths.shape != (x.shape[0],) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError('lengths must be an integer vector, one per sequence')
        if torch.any(lengths < 1) or torch.any(lengths > x.shape[1]):
            raise ValueError('lengths must be within the supplied sequence dimension')
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths.to(x.device)[:, None]
        # Exclude padding BEFORE BatchNorm as well as before the RNN.
        encoded = self.encode_windows(x[mask])
        features = encoded.new_zeros(x.shape[0], x.shape[1], encoded.shape[-1])
        features[mask] = encoded
        return self.classify_features(features, lengths)
