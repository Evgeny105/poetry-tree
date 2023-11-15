import csv
import pickle

import hydra
from hydra.core.config_store import ConfigStore
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


try:
    from .config import Params
except ImportError:
    from poetry_tree.config import Params

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(
    version_base="1.3.2", config_path="..\\config", config_name="config"
)
def main(cfg: Params):
    RANDOM_STATE = cfg.train.random_state

    digits_data = load_digits().data
    digits_target = load_digits().target[:, None]
    X_train, X_test, y_train, y_test = train_test_split(
        digits_data, digits_target, test_size=0.2, random_state=RANDOM_STATE
    )

    with open(cfg.model.path, "rb") as f:
        class_estimator = pickle.load(f)

    answer = class_estimator.predict(X_test)
    accuracy = accuracy_score(y_test, answer)
    print(
        "Accuracy of tree with criterion '" + cfg.infer.criterion_name + "' is",
        accuracy,
    )

    with open(cfg.infer.result_csv_path, "w", newline="") as csvfile:
        fieldnames = ["reference", "answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ref, ans in zip(y_test, answer):
            writer.writerow({"reference": ref[0], "answer": ans})


if __name__ == "__main__":
    main()
