# import math
# import pickle
# import random
# import sys
# import time
# from datetime import datetime

# import git
import hydra

# import mlflow
# import mlflow.onnx
# import numpy as np
import onnx

# import onnxruntime
# import spacy
import torch

# import torch.nn as nn
import torch.onnx

# import torchdata.datapipes as dp
# import torchtext.transforms as T
from hydra.core.config_store import ConfigStore

from poetry_tree.config import Params


# from spacy_download import load_spacy
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader, Sampler
# from torchtext.vocab import build_vocab_from_iterator


# from poetry_tree.transformer import Decoder, Encoder, Seq2Seq


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(version_base="1.3.2", config_path="../config", config_name="config")
def main(cfg: Params):
    import onnxruntime

    # onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
    # print(f"Input length: {len(onnx_input)}")
    # print(f"Sample input: {onnx_input}")

    ort_session = onnxruntime.InferenceSession(
        "mlartifacts/883917725595293284/97990527985342b880f848ff7a4d40e5/artifacts/models",
        providers=["CPUExecutionProvider"],
    )

    # def to_numpy(tensor):
    #     return (
    #         tensor.detach().cpu().numpy()
    #         if tensor.requires_grad
    #         else tensor.cpu().numpy()
    #     )

    # onnxruntime_input = {
    #     k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), device)
    # }

    # onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Путь к модели в MLflow (замените на фактический путь)
    # Загрузка модели из MLflow
    # onnx_model = mlflow.onnx.load_model(
    #     "mlartifacts/883917725595293284/97990527985342b880f848ff7a4d40e5/artifacts/models"
    # )
    onnx_model = onnx.load(cfg.model.path_onnx)
    # Создание сессии ONNX Runtime
    ort_session = onnxruntime.InferenceSession(onnx_model)

    # Пример входных данных для инференса
    input_onnx_src = torch.randint(
        0, 100, (1, 50), device=device, dtype=torch.int
    )
    input_onnx_trg = torch.randint(
        0, 100, (1, 50), device=device, dtype=torch.int
    )

    # Инференс с помощью модели
    outputs, attention = ort_session.run(
        None, {"source": input_onnx_src, "target": input_onnx_trg}
    )

    # Вывод результата
    print("Model output:", outputs, attention)

    # model.load_state_dict(torch.load(cfg.model.path))
    # model.eval()
    # input_onnx_src = torch.randint(
    #     0, INPUT_DIM, (1, 50), device=device, dtype=torch.int
    # )
    # input_onnx_trg = torch.randint(
    #     0, OUTPUT_DIM, (1, 50), device=device, dtype=torch.int
    # )
    # example_output, example_attention = model(input_onnx_src, input_onnx_trg)
    # torch.onnx.export(
    #     model,
    #     args=(input_onnx_src, input_onnx_trg),
    #     f=cfg.model.path_onnx,
    #     verbose=True,
    #     input_names=["source", "target"],
    #     output_names=["output", "attention"],
    # )
    # onnx_model = onnx.load(cfg.model.path_onnx)
    # mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path="models")


if __name__ == "__main__":
    main()
