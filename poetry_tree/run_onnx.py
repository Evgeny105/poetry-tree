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
import dvc.api
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
    with dvc.api.open(cfg.model.path_onnx, mode="rb") as f:
        ort_session = onnxruntime.InferenceSession(
            f.name,
            providers=["CPUExecutionProvider"],
        )

    input_onnx_src = torch.zeros(1, 50, dtype=torch.int)
    # фраза для перевода:
    # <sos> предоставляются полотенца . <eos>
    input_onnx_src[0, :5] = torch.tensor([2, 81, 219, 4, 3], dtype=torch.int)

    input_onnx_trg = torch.zeros(1, 50, dtype=torch.int)
    # стартовый токен для начала генерации:
    # <sos>
    input_onnx_trg[0, :1] = torch.tensor([2], dtype=torch.int)

    outputs, _ = ort_session.run(
        None,
        {
            "source": input_onnx_src.numpy(),
            "target": input_onnx_trg.numpy(),
        },
    )
    # with dvc.api.open(cfg.model.path_to_eng_vocab, mode="rb") as f:
    #     eng_vocab = pickle.load(f)
    print("Model output:", outputs)


if __name__ == "__main__":
    main()
