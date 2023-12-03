import fire
from hydra import compose, initialize

from poetry_tree.infer import main as infer_py
from poetry_tree.train import main as train_py


def train():
    with initialize(version_base="1.3.2", config_path="./config"):
        cfg = compose(config_name="config")
    train_py(cfg)


def infer():
    with initialize(version_base="1.3.2", config_path="./config"):
        cfg = compose(config_name="config")
    infer_py(cfg)


def main():
    fire.Fire(
        {
            "infer": infer,
            "train": train,
        }
    )


if __name__ == "__main__":
    main()
