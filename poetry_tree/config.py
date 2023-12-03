from dataclasses import dataclass
from typing import Optional


@dataclass
class Model:
    path: str
    path_onnx: str
    path_to_rus_vocab: str
    path_to_eng_vocab: str
    hid_dim: int
    enc_layers: int
    dec_layers: int
    enc_heads: int
    dec_heads: int
    enc_pf_dim: int
    dec_pf_dim: int
    enc_dropout: float
    dec_dropout: float


@dataclass
class Train:
    mlflow_uri: str
    random_state: Optional[int]
    path_to_dataset: str
    batch_size: int
    learning_rate: float
    n_epochs: int
    clip_grad: float
    # criterion_name: str
    # max_depth: int


@dataclass
class Infer:
    random_state: Optional[int]
    # result_csv_path: str
    # criterion_name: str


@dataclass
class Params:
    model: Model
    train: Train
    infer: Infer
