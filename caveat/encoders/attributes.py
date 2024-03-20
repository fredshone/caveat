import pandas as pd
import torch
from torch import Tensor


class AttributeEncoder:
    def __init__(self, config: dict) -> None:
        self.config = config

    def encode(self, data: pd.DataFrame) -> Tensor:
        encoded = []
        for k, v in self.config.items():
            if k not in data.columns:
                raise UserWarning(f"Conditional '{k}' not found in attributes")

            if isinstance(v, dict) and isinstance(v.get("ordinal"), tuple):
                min, max = v["ordinal"]
                encoded.append(ordinal_encode(data[k], min, max))

            elif v == "nominal":
                encoded.append(nominal_encode(data[k]))

            else:
                raise UserWarning(
                    f"Unrecognised attribute encoding in configuration: {v}"
                )

        if not encoded:
            raise UserWarning("No attributes encoded.")

        return torch.cat(encoded, dim=-1)


def ordinal_encode(data: pd.Series, min, max) -> Tensor:
    encoded = Tensor(data.values).unsqueeze(-1)
    encoded -= min
    encoded /= max - min
    return encoded.float()


def nominal_encode(data: pd.Series) -> Tensor:
    nominals = pd.Categorical(data)
    encodings = nominals.categories
    nominals = Tensor(nominals.codes).long()
    nominals = torch.nn.functional.one_hot(
        nominals, num_classes=len(encodings)
    ).float()
    return nominals, encodings
