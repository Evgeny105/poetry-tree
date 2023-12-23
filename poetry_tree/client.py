import pickle
from functools import lru_cache

import dvc.api
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)
from tritonclient.utils import np_to_triton_dtype


# from poetry_tree.config import Params
try:
    from poetry_tree.config import Params
except ImportError:
    from config import Params


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


@hydra.main(version_base="1.3.2", config_path="../config", config_name="config")
def main(cfg: Params):
    triton_client = get_client()

    with dvc.api.open(cfg.model.path_to_eng_vocab, mode="rb") as f:
        eng_vocab = pickle.load(f)

    input_onnx_src = np.ones((1, 50), dtype=np.int32)
    # фраза для перевода:
    print("Request for model: \n<sos> предоставляются полотенца . <eos>")
    input_onnx_src[0, :5] = np.array([2, 81, 219, 4, 3], dtype=np.int32)

    input_onnx_trg = np.ones((1, 50), dtype=np.int32)
    # стартовый токен для начала генерации:
    # <sos>
    input_onnx_trg[0, 0] = eng_vocab["<sos>"]

    input_source = InferInput(
        name="source",
        shape=input_onnx_src.shape,
        datatype=np_to_triton_dtype(input_onnx_src.dtype),
    )
    input_target = InferInput(
        name="target",
        shape=input_onnx_trg.shape,
        datatype=np_to_triton_dtype(input_onnx_trg.dtype),
    )

    input_source.set_data_from_numpy(input_onnx_src, binary_data=True)
    infer_output = InferRequestedOutput("output", binary_data=True)

    print("Answer of model:")
    words = ["<sos>"]

    for i in range(50):
        input_target.set_data_from_numpy(input_onnx_trg, binary_data=True)
        query_response = triton_client.infer(
            "onnx-model", [input_source, input_target], outputs=[infer_output]
        )
        output = query_response.as_numpy("output")
        token = np.argmax(output[0][i])
        words.append(eng_vocab.get_itos()[token])

        if token == eng_vocab["<eos>"]:
            break

        input_onnx_trg[0, i + 1] = token
    print(" ".join(words))


if __name__ == "__main__":
    main()

"""
Request for model:
<sos> предоставляются полотенца . <eos>
Answer of model:
<sos> towels are provided . <eos>
"""
