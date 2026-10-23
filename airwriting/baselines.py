"""Re-implementations of published trajectory recognizers for comparison.

Each class follows the architecture reported in the original paper. Details
the papers leave unspecified are marked "assumption" and kept minimal. All
models consume the concatenated feature branches of shape (batch, time, dim).

- Carbune et al., "Fast multi-language LSTM-based online handwriting
  recognition", IJDAR 2020: stacked BiLSTM, 64 cells, dropout 0.5, CTC.
- Ott et al., "Benchmarking online sequence-to-sequence and character-based
  handwriting recognition from IMU-enhanced pens", IJDAR 2022: CNN+BiLSTM, CTC.
- Alam et al., "Trajectory-based air-writing recognition using deep neural
  network and depth sensor", Sensors 2020: LSTM and CNN closed-set classifiers.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CarbuneBiLSTM(nn.Module):
    """Five BiLSTM layers with 64 cells per direction and dropout 0.5 after
    each layer, followed by a softmax over characters plus the CTC blank."""

    task = "ctc"

    def __init__(self, input_dim: int, num_classes: int, layers: int = 5, hidden: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.LSTM(input_dim if index == 0 else 2 * hidden, hidden, batch_first=True, bidirectional=True)
                for index in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(2 * hidden, num_classes)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        values = inputs
        for layer in self.layers:
            values, _ = layer(values)
            values = self.dropout(values)
        log_probs = F.log_softmax(self.output(values), dim=-1).transpose(0, 1)
        return log_probs, lengths, None


class OttCNNBiLSTM(nn.Module):
    """Conv1D (200 filters, kernel 4) -> max pooling (2) -> batch norm ->
    dropout 0.2 -> two BiLSTM layers (60 units, tanh) -> dense -> CTC.

    Assumption: ReLU after the convolution (the activation is not stated) and
    'same' padding. The dense output has one unit per CTC symbol.
    """

    task = "ctc"

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(input_dim, 200, 4, padding="same")
        self.pool = nn.MaxPool1d(2)
        self.norm = nn.BatchNorm1d(200)
        self.dropout = nn.Dropout(0.2)
        self.recurrent = nn.LSTM(200, 60, num_layers=2, batch_first=True, bidirectional=True)
        self.output = nn.Linear(120, num_classes)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        values = F.relu(self.convolution(inputs.transpose(1, 2)))
        values = self.dropout(self.norm(self.pool(values))).transpose(1, 2)
        values, _ = self.recurrent(values)
        log_probs = F.log_softmax(self.output(values), dim=-1).transpose(0, 1)
        return log_probs, torch.div(lengths, 2, rounding_mode="floor"), None


class AlamLSTMClassifier(nn.Module):
    """LSTM (64) -> LSTM (128) -> dense 64 -> dense 256 (ReLU, dropout 0.5)
    -> softmax over the closed set of training labels."""

    task = "classification"

    def __init__(self, input_dim: int, num_labels: int) -> None:
        super().__init__()
        self.first = nn.LSTM(input_dim, 64, batch_first=True)
        self.second = nn.LSTM(64, 128, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_labels),
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        values, _ = self.first(inputs)
        _, (hidden, _) = self.second(values)
        return self.head(hidden[-1])


class AlamCNNClassifier(nn.Module):
    """Three depthwise-separable Conv1D layers (64, 128, 256 channels, kernel 3),
    each followed by max pooling, then dense 256 and 128 (ReLU, dropout 0.5)
    and a softmax over the training labels.

    Assumption: ReLU after each convolution, pool size 2, 'same' padding.
    """

    task = "classification"

    def __init__(self, input_dim: int, num_labels: int, length: int = 128) -> None:
        super().__init__()
        layers = []
        channels = input_dim
        for width in (64, 128, 256):
            layers += [
                nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
                nn.Conv1d(channels, width, 1),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (length // 8), 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_labels),
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        return self.head(self.features(inputs.transpose(1, 2)))


BASELINES = {
    "carbune_bilstm": CarbuneBiLSTM,
    "ott_cnn_bilstm": OttCNNBiLSTM,
    "alam_lstm": AlamLSTMClassifier,
    "alam_cnn": AlamCNNClassifier,
}


class BaselineModel(nn.Module):
    """Concatenates feature branches in a fixed order and runs a baseline."""

    def __init__(self, backbone: str, branch_widths: dict[str, int], outputs: int, length: int) -> None:
        super().__init__()
        self.names = list(branch_widths)
        width = sum(branch_widths.values())
        if backbone == "alam_cnn":
            self.network = AlamCNNClassifier(width, outputs, length)
        else:
            self.network = BASELINES[backbone](width, outputs)
        self.task = self.network.task

    def forward(self, branches: dict[str, torch.Tensor], lengths: torch.Tensor):
        inputs = torch.cat([branches[name] for name in self.names], dim=-1)
        return self.network(inputs, lengths)
