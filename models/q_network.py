import torch
import torch.nn as nn

from config import ROWS, COLS


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = ROWS * COLS
        self.hidden_dim = 128
        self.output_dim = COLS

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):
        return self.network(x)