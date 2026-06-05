import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_loader import load_batadal
from preprocessing.data_splitter import split_batadal

from models.automata.automata import ProbabilisticAutomata
from explainability.explainability import AutomataExplainer


def main():

    batadal = load_batadal()

    result = split_batadal(batadal)

    X_train = result[3]
    X_test = result[5]

    y_train = result[6]

    model = ProbabilisticAutomata()

    model.fit(X_train, y_train)

    explainer = AutomataExplainer(model)

    explanation = explainer.explain_window(
        X_test[:100],
        time_step=0
    )

    print(explainer.yazili_acikla(explanation))


if __name__ == "__main__":
    main()