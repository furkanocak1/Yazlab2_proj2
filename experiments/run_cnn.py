import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
from preprocessing.data_loader import load_batadal
from preprocessing.data_splitter import split_batadal
from models.deep_learning.cnn_model import CNNModel


def to_cnn_tensor(X):
    return torch.tensor(X, dtype=torch.float32).unsqueeze(1)


def main():
    batadal = load_batadal()
    result = split_batadal(batadal)

    X_train, X_val, X_test = result[0], result[1], result[2]
    y_train, y_val, y_test = result[6], result[7], result[8]

    X_train = to_cnn_tensor(X_train)
    X_test = to_cnn_tensor(X_test)

    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    input_size = X_train.shape[2]
    model = CNNModel(input_size=input_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("CNN training started")

    for epoch in range(config.EPOCH_LIMIT):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.EPOCH_LIMIT} - Loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predicted_labels = (predictions >= 0.5).int().numpy()

    y_true = y_test_tensor.int().numpy()

    print("\nCNN Test Sonuclari")
    print("Accuracy :", accuracy_score(y_true, predicted_labels))
    print("Precision:", precision_score(y_true, predicted_labels, zero_division=0))
    print("Recall   :", recall_score(y_true, predicted_labels, zero_division=0))
    print("F1 Score :", f1_score(y_true, predicted_labels, zero_division=0))


if __name__ == "__main__":
    main()