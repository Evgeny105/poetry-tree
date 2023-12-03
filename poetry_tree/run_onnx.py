import pickle

import dvc.api
import hydra
import numpy as np
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

    input_onnx_src = torch.ones(1, 50, dtype=torch.int)
    # фраза для перевода:
    print("Request for model: \n<sos> предоставляются полотенца . <eos>")
    input_onnx_src[0, :5] = torch.tensor([2, 81, 219, 4, 3], dtype=torch.int)

    input_onnx_trg = torch.ones(1, 50, dtype=torch.int)
    # стартовый токен для начала генерации:
    # <sos>
    input_onnx_trg[0, 0] = torch.tensor([2], dtype=torch.int)

    with dvc.api.open(cfg.model.path_to_eng_vocab, mode="rb") as f:
        eng_vocab = pickle.load(f)

    print("Answer of model:")
    words = ["<sos>"]
    for i in range(50):
        outputs, _ = ort_session.run(
            None,
            {
                "source": input_onnx_src.numpy(),
                "target": input_onnx_trg.numpy(),
            },
        )
        token = np.argmax(outputs[0, i, :])
        words.append(eng_vocab.get_itos()[token])

        if token == eng_vocab["<eos>"]:
            break

        input_onnx_trg[0, i + 1] = torch.tensor([token], dtype=torch.int)
    print(" ".join(words))


"""
Request for model:
<sos> предоставляются полотенца . <eos>
Answer of model:
<sos> towels are provided . <eos>
"""


if __name__ == "__main__":
    main()
