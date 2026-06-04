import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deep_learning.lstm_model import LSTMModel


def main():
    model = LSTMModel(input_size=10)
    print(model)
    print("LSTM experiment started")


if __name__ == "__main__":
    main()