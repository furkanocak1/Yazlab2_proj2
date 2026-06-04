import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deep_learning.gru_model import GRUModel


def main():
    model = GRUModel(input_size=10)
    print(model)
    print("GRU experiment started")


if __name__ == "__main__":
    main()