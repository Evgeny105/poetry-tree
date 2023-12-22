import pickle
from functools import lru_cache

import numpy as np

# from transformers import AutoTokenizer
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)
from tritonclient.utils import np_to_triton_dtype


# import spacy
# import torch
# import torchtext.transforms as T


# def get_transform(vocab):
#     text_transform = T.Sequential(
#         T.VocabTransform(vocab=vocab),
#         T.AddToken(vocab["<sos>"], begin=True),
#         T.AddToken(vocab["<eos>"], begin=False),
#     )
#     return text_transform


# eng = spacy.blank("en").from_disk("models/eng_tokenizer")

with open("models/eng_vocab.pickle", mode="rb") as f:
    eng_vocab = pickle.load(f)
# words_target = [token.text for token in eng.tokenizer("")]
# # words_source = [token.text for token in rus.tokenizer("Пляж рядом с отелем")]
# # tokens_rus = get_transform(rus_vocab)(words_source)
# if len(words_target) == 0:
#     tokens_eng = [
#         eng_vocab["<sos>"],
#     ]
# else:
#     tokens_eng = get_transform(eng_vocab)(words_target)[:-1]
# # tokens_rus = tokens_rus[:50] + [
# #     self.rus_vocab["<pad>"]
# #     for _ in range(50 - len(tokens_rus[:50]))
# # ]
# tokens_eng = tokens_eng[:50] + [
#     eng_vocab["<pad>"] for _ in range(50 - len(tokens_eng[:50]))
# ]


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


# ENSEMBLE
# text_source = "пляж рядом с отелем"
# text_target = ""

# triton_client = get_client()

# text_source = np.array([text_source.encode("utf-8")], dtype=object)
# text_target = np.array([text_target.encode("utf-8")], dtype=object)
# input_text_s = InferInput(
#     name="TEXTS_SOURCE",
#     shape=text_source.shape,
#     datatype=np_to_triton_dtype(text_source.dtype),
# )
# input_text_t = InferInput(
#     name="TEXTS_TARGET",
#     shape=text_target.shape,
#     datatype=np_to_triton_dtype(text_target.dtype),
# )
# input_text_s.set_data_from_numpy(text_source, binary_data=True)
# input_text_t.set_data_from_numpy(text_target, binary_data=True)

# infer_output = InferRequestedOutput("output", binary_data=True)
# query_response = triton_client.infer(
#     "ensemble-onnx", [input_text_s, input_text_t], outputs=[infer_output]
# )
# output = query_response.as_numpy("output")
# tokens = np.argmax(output, 1)
# words = [eng_vocab.get_itos()[t] for t in tokens]
# print(" ".join(words))


# MODEL
triton_client = get_client()

input_onnx_src = np.ones((1, 50), dtype=np.int32)
# фраза для перевода:
# <sos> предоставляются полотенца . <eos>
input_onnx_src[0, :5] = np.array([2, 81, 219, 4, 3], dtype=np.int32)

input_onnx_trg = np.ones((1, 50), dtype=np.int32)
# стартовый токен для начала генерации:
# <sos>
input_onnx_trg[0, :3] = np.array([2, 202, 17], dtype=np.int32)

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
input_target.set_data_from_numpy(input_onnx_trg, binary_data=True)

infer_output = InferRequestedOutput("output", binary_data=True)
query_response = triton_client.infer(
    "onnx-model", [input_source, input_target], outputs=[infer_output]
)
output = query_response.as_numpy("output")
tokens = np.argmax(output[0], 1)
words = [eng_vocab.get_itos()[t] for t in tokens]
print(" ".join(words))
