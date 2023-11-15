from dataclasses import dataclass
from typing import Optional


@dataclass
class Model:
    path: str


@dataclass
class Train:
    criterion_name: str
    max_depth: int
    random_state: Optional[int]


@dataclass
class Infer:
    result_csv_path: str
    criterion_name: str


@dataclass
class Params:
    model: Model
    train: Train
    infer: Infer
