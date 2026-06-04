import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deep_learning.cnn_model import CNNModel


def main():
    model = CNNModel(input_size=10)
    print(model)
    print("CNN experiment started")


if __name__ == "__main__":
    main()  