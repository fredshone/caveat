from typing import Optional

import pandas as pd
import torch
from torch import Tensor


class AttributeEncoder:
    def __init__(self, config: dict) -> None:
        self.config = config

    def encode(self, data: pd.DataFrame) -> Tensor:
        encoded = []
        for k, v in self.config.copy().items():
            if k not in data.columns:
                raise UserWarning(f"Conditional '{k}' not found in attributes")

            if isinstance(v, dict):  # Defined encoding
                if v.get("ordinal"):
                    if not isinstance(v.get("ordinal"), tuple):
                        raise UserWarning(
                            f"Ordinal encoding must be a tuple of (min, max)"
                        )
                    min, max = v["ordinal"]
                    encoded.append(ordinal_encode(data[k], min, max))
                elif v.get("nominal"):
                    if not isinstance(v.get("nominal"), dict):
                        raise UserWarning(
                            f"Nominal encoding must be a dict of categories to index"
                        )
                    nominal_encoded, _ = nominal_encode(
                        data[k], v["nominal"]
                    )
                    encoded.append(nominal_encoded)
                else:
                    raise UserWarning(
                        f"Unrecognised attribute encoding in configuration: {v}"
                    )

            elif v == "nominal":  # Undefined nominal encoding
                nominal_encoded, nominal_encodings = nominal_encode(data[k], None)
                self.config[k] = {"nominal": nominal_encodings}
                encoded.append(nominal_encoded)

            else:
                raise UserWarning(
                    f"Unrecognised attribute encoding in configuration: {v}"
                )

        if not encoded:
            raise UserWarning("No attribute encodeding found.")

        return torch.cat(encoded, dim=-1)


def ordinal_encode(data: pd.Series, min, max) -> Tensor:
    encoded = Tensor(data.values).unsqueeze(-1)
    encoded -= min
    encoded /= max - min
    return encoded.float()


def nominal_encode(data: pd.Series, encodings: Optional[dict] = None) -> Tensor:
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
    nominals = Tensor(nominals.codes).long()
    nominals = torch.nn.functional.one_hot(
        nominals, num_classes=len(encodings)
    ).float()
    return nominals, encodings
