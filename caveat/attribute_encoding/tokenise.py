import pandas as pd
import torch
from torch import Tensor

from caveat.attribute_encoding.base import BaseAttributeEncoder, tokenize


class TokenAttributeEncoder(BaseAttributeEncoder):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.sizes = []

    def encode(self, data: pd.DataFrame) -> Tensor:
        encoded = []
        for i, (k, v) in enumerate(self.config.copy().items()):
            if k not in data.columns:
                raise UserWarning(f"Conditional '{k}' not found in attributes")
            if isinstance(v, dict):  # Pre-defined encoding
                if v.get("ordinal"):
                    raise UserWarning(
                        "Ordinal encoding not supported for token encoding, change config to nominal or remove"
                    )
                elif v.get("nominal"):
                    if not isinstance(v.get("nominal"), dict):
                        raise UserWarning(
                            "Nominal encoding must be a dict of categories to index"
                        )
                    nominal_encoded, _ = tokenize(data[k], v["nominal"])
                    encoded.append(nominal_encoded)
                    self.config[k].update(
                        {
                            "location": i,
                            # "size": len(v["nominal"]),
                            "type": data[k].dtype,
                        }
                    )
                else:
                    raise UserWarning(
                        f"Unrecognised attribute encoding in configuration: {v}"
                    )

            elif v == "nominal":  # Undefined nominal encoding
                nominal_encoded, nominal_encodings = tokenize(data[k], None)
                encoded.append(nominal_encoded)
                self.config[k] = {
                    "nominal": nominal_encodings,
                    "location": i,
                    # "size": len(nominal_encodings),
                    "type": data[k].dtype,
                }
                self.sizes.append(len(nominal_encodings))

            elif v == "ordinal":  # Undefined ordinal encoding
                raise UserWarning(
                    "Ordinal encoding not supported for token encoding, change config to nominal or remove"
                )

            else:
                raise UserWarning(
                    f"Unrecognised attribute encoding in configuration: {v}"
                )

        if not encoded:
            raise UserWarning("No attribute encoding found.")

        return torch.stack(encoded, dim=-1)

    def decode(self, data: Tensor) -> pd.DataFrame:
        decoded = {"pid": list(range(data.shape[0]))}
        for k, v in self.config.items():
            location, column_type = (v["location"], v["type"])
            if v.get("nominal") is not None:
                encoding = {i: name for name, i in v["nominal"].items()}
                decoded[k] = pd.Series(
                    [
                        encoding[i]
                        for i in data[:, location].argmax(dim=-1).tolist()
                    ]
                ).astype(column_type)
            else:
                raise UserWarning(
                    f"Unrecognised attribute encoding in configuration: {k, v}"
                )

        return pd.DataFrame(decoded)
