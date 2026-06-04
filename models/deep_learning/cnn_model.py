import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_size=3, output_size=1, dropout=0.2):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return self.sigmoid(x)