from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from caveat.encoders import BaseEncodedPlans, BaseEncoder


class SequenceEncoder(BaseEncoder):
    def __init__(
        self, max_length: int = 12, norm_duration: int = 1440, **kwargs
    ):
        self.max_length = max_length
        self.norm_duration = norm_duration

    def encode(self, data: pd.DataFrame) -> BaseEncodedPlans:
        self.sos = 0
        self.eos = 1
        self.index_to_acts = {i + 2: a for i, a in enumerate(data.act.unique())}
        self.index_to_acts[0] = "<SOS>"
        self.index_to_acts[1] = "<EOS>"
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}
        self.encodings = len(self.index_to_acts)
        # encoding takes place in SequenceDataset
        return SequenceEncodedPlans(
            data, self.max_length, self.acts_to_index, self.norm_duration
        )

    def decode(self, encoded: Tensor) -> pd.DataFrame:
        """Decode a sequences ([N, max_length, encoding]) into DataFrame of 'traces', eg:

        pid | act | start | end

        enumeration of seq is used for pid.

        Args:
            encoded (Tensor): _description_

        Returns:
            pd.DataFrame: _description_
        """
        encoded, durations = torch.split(encoded, [self.encodings, 1], dim=-1)
        encoded = encoded.argmax(dim=-1).numpy()
        decoded = []

        for pid in range(len(encoded)):
            act_start = 0
            for act_idx, duration in zip(encoded[pid], durations[pid]):
                if int(act_idx) == self.sos:
                    continue
                if int(act_idx) == self.eos:
                    break
                duration = int(duration * self.norm_duration)
                decoded.append(
                    [
                        pid,
                        self.index_to_acts[int(act_idx)],
                        act_start,
                        act_start + duration,
                    ]
                )
                act_start += duration

        return pd.DataFrame(decoded, columns=["pid", "act", "start", "end"])


class SequenceEncodedPlans(BaseEncodedPlans):
    def __init__(
        self,
        data: pd.DataFrame,
        max_length: int,
        acts_to_index: dict,
        norm_duration: int,
        sos: int = 0,
        eos: int = 1,
    ):
        """Torch Dataset for sequence data.

        Args:
            data (DataFrame): Population of sequences.
            max_length (int): Max length of sequences.
            acts_to_index (dict): Mapping of activity to index.
        """
        self.max_length = max_length
        self.sos = sos
        self.eos = eos
        self.encodings = len(acts_to_index)
        self.encoded, self.masks, self.encoding_weights = self._encode(
            data, max_length, acts_to_index, norm_duration
        )
        self.size = len(self.encoded)

    def _encode(
        self,
        data: pd.DataFrame,
        max_length: int,
        acts_to_index: dict,
        norm_duration: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        data = data.copy()
        data.act = data.act.map(acts_to_index)

        # calc weightings
        weights = data.groupby("act", observed=True).duration.sum().to_dict()
        n = (
            data.pid.nunique() * 60
        )  # sos and eos weight is equal to 1 hour per sequence
        weights.update({self.sos: n, self.eos: n})
        weights = np.array([weights[k] for k in range(len(weights))])
        weights = 1 / weights

        data.duration = data.duration / norm_duration
        persons = data.pid.nunique()
        encoding_width = 2  # cat act encoding plus duration

        encoded = np.zeros(
            (persons, max_length, encoding_width), dtype=np.float32
        )
        masks = np.zeros((persons, max_length), dtype=np.int8)

        for pid, (_, trace) in enumerate(data.groupby("pid")):
            encoding, mask = encode_sequence(
                acts=list(trace.act),
                durations=list(trace.duration),
                max_length=max_length,
                encoding_width=encoding_width,
                sos=self.sos,
                eos=self.eos,
            )
            encoded[pid] = encoding
            masks[pid] = mask
            # [N, L, W]

        return (
            torch.from_numpy(encoded),
            torch.from_numpy(masks),
            torch.from_numpy(weights).float(),
        )

    def shape(self):
        return self.encoded[0].shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.encoded[idx], self.masks[idx]


def encode_sequence(
    acts: list[int],
    durations: list[float],
    max_length: int,
    encoding_width: int,
    sos: int,
    eos: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequence encoding from ranges.

    Args:
        acts (Iterable[int]): _description_
        durations (Iterable[float]): _description_
        max_length (int): _description_
        encoding_width (dict): _description_

    Returns:
        np.array: _description_
    """
    encoding = np.zeros((max_length, encoding_width), dtype=np.float32)
    mask = np.zeros((max_length))
    # SOS
    encoding[0][0] = sos
    # mask includes sos
    mask[0] = 1
    for i in range(1, max_length):
        if i < len(acts) + 1:
            encoding[i][0] = acts[i - 1]
            encoding[i][1] = durations[i - 1]
            mask[i] = 1
        elif i < len(acts) + 2:
            encoding[i][0] = eos
            # mask includes first eos
            mask[i] = 1
        else:
            encoding[i][0] = eos
    return encoding, mask
