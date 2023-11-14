# import logging
import pickle

import hydra
from omegaconf import DictConfig
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


try:
    from .tree import DecisionTree
except ImportError:
    from poetry_tree.tree import DecisionTree


@hydra.main(
    version_base="1.3.2", config_path="..\\config", config_name="config"
)
def main(cfg: DictConfig):
    RANDOM_STATE = cfg.train.random_state

    digits_data = load_digits().data
    digits_target = load_digits().target[:, None]
    X_train, X_test, y_train, y_test = train_test_split(
        digits_data, digits_target, test_size=0.2, random_state=RANDOM_STATE
    )

    class_estimator = DecisionTree(
        max_depth=cfg.train.max_depth, criterion_name=cfg.train.criterion_name
    )
    print("fit...")
    class_estimator.fit(X_train, y_train)

    with open(cfg.model.path, "wb") as f:
        pickle.dump(class_estimator, f)


if __name__ == "__main__":
    main()
