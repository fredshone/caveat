import pandas as pd
import torch
from torch import Tensor

from caveat.attribute_encoding.base import BaseLabelEncoder, tokenize


class TokenAttributeEncoder(BaseLabelEncoder):

    def encode(self, data: pd.DataFrame) -> Tensor:
        encoded = []
        self.label_kwargs["attribute_embed_sizes"] = []

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
                    nominal_encoded, nominal_encodings = tokenize(
                        data[k], v["nominal"]
                    )
                    encoded.append(nominal_encoded)
                    self.config[k].update(
                        {"location": i, "type": data[k].dtype}
                    )
                    self.label_kwargs["attribute_embed_sizes"].append(
                        len(nominal_encodings)
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
                    "type": data[k].dtype,
                }
                self.label_kwargs["attribute_embed_sizes"].append(
                    len(nominal_encodings)
                )

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

        return torch.stack(encoded, dim=-1).long()

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
