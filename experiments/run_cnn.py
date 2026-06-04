import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_loader import load_batadal
from preprocessing.data_splitter import split_batadal
from models.deep_learning.cnn_model import CNNModel


def main():
    batadal = load_batadal()
    result = split_batadal(batadal)

    X_train, X_val, X_test = result[0], result[1], result[2]
    y_train, y_val, y_test = result[6], result[7], result[8]

    input_size = X_train.shape[1]

    model = CNNModel(input_size=input_size)

    print(model)
    print("BATADAL data loaded")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("CNN experiment ready")


if __name__ == "__main__":
    main()