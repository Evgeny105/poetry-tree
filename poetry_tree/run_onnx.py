# import math
# import pickle
# import random
# import sys
# import time
# from datetime import datetime
# import git
# import mlflow
# import mlflow.onnx
# import numpy as np
# import onnx
# import onnxruntime
# import spacy
# import torch.nn as nn
# import torchdata.datapipes as dp
# import torchtext.transforms as T
# from spacy_download import load_spacy
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader, Sampler
# from torchtext.vocab import build_vocab_from_iterator
# from poetry_tree.transformer import Decoder, Encoder, Seq2Seq
import hydra
import onnxruntime
import torch
import torch.onnx
from hydra.core.config_store import ConfigStore

from poetry_tree.config import Params


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(version_base="1.3.2", config_path="../config", config_name="config")
def main(cfg: Params):
    ort_session = onnxruntime.InferenceSession(
        cfg.model.path_onnx,
        providers=["CPUExecutionProvider"],
    )

    input_onnx_src = torch.randint(0, 100, (1, 50), dtype=torch.int)
    input_onnx_trg = torch.randint(0, 100, (1, 50), dtype=torch.int)

    outputs, attention = ort_session.run(
        None,
        {
            "source": input_onnx_src.cpu().numpy(),
            "target": input_onnx_trg.cpu().numpy(),
        },
    )

    print("Model output:", outputs, attention)


if __name__ == "__main__":
    main()
