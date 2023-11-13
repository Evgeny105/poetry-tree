import pickle

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .tree import DecisionTree


def main():
    RANDOM_STATE = 42

    digits_data = load_digits().data
    digits_target = load_digits().target[:, None]
    X_train, X_test, y_train, y_test = train_test_split(
        digits_data, digits_target, test_size=0.2, random_state=RANDOM_STATE
    )

    class_estimator = DecisionTree(max_depth=10, criterion_name="gini")
    print("fit...")
    class_estimator.fit(X_train, y_train)

    with open("data.pickle", "wb") as f:
        pickle.dump(class_estimator, f)


if __name__ == "__main__":
    main()
