from typing import Optional

import pandas as pd
import torch
from torch import Tensor


class BaseLabelEncoder:
    def __init__(self, config: dict) -> None:
        """Base Attribute Encoder class."""
        self.config = config
        self.label_kwargs = {}

    def encode(self, data: pd.DataFrame) -> Tensor:
        raise NotImplementedError

    def decode(self, data: Tensor) -> pd.DataFrame:
        raise NotImplementedError


def ordinal_encode(data: pd.Series, min, max) -> Tensor:
    encoded = Tensor(data.values).unsqueeze(-1)
    encoded -= min
    encoded /= max - min
    return encoded.float()


def tokenize(data: pd.Series, encodings: Optional[dict] = None) -> Tensor:
    if encodings:
        missing = set(data.unique()) - set(encodings.keys())
        if missing:
            raise UserWarning(
                f"""
                Categories in data do not match existing categories: {missing}.
                Please specify the new categories in the encoding.
                Your existing encodings are: {encodings}
"""
            )
        nominals = pd.Categorical(data, categories=encodings.keys())
    else:
        nominals = pd.Categorical(data)
        encodings = {e: i for i, e in enumerate(nominals.categories)}
    nominals = torch.tensor(nominals.codes).int()
    return nominals, encodings


def onehot_encode(data: pd.Series, encodings: Optional[dict] = None) -> Tensor:
    nominals, encodings = tokenize(data, encodings)
    nominals = torch.nn.functional.one_hot(
        nominals.long(), num_classes=len(encodings)
    ).float()
    return nominals, encodings
