"""
Iris species classification with a support vector machine.

Trains an SVM on the Iris dataset and reports accuracy on a held-out test split.
The dataset is 150 samples across three species, with four measurements each.

    pip install pandas scikit-learn
    python iris_svm.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABEL = "species"

# A fixed seed keeps the split reproducible. Without it, accuracy moves by a few
# points between runs purely because a different 30 samples land in the test set,
# which makes the number impossible to compare against anything.
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    df = pd.read_csv("iris.csv")

    X = df[FEATURES]
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = SVC()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Test accuracy:    {accuracy:.3f}")


if __name__ == "__main__":
    main()
