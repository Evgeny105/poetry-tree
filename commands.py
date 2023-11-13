import fire

from poetry_tree.infer import main as infer
from poetry_tree.train import main as train


if __name__ == "__main__":
    fire.Fire(
        {
            "infer": infer,
            "train": train,
        }
    )
