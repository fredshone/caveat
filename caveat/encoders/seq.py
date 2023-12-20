import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from caveat.encoders import BaseEncoder


class Sequence(BaseEncoder):
    def __init__(
        self, max_length: int = 12, norm_duration: int = 1440, **kwargs
    ):
        self.max_length = max_length
        self.norm_duration = norm_duration

    def encode(self, data: pd.DataFrame) -> Dataset:
        self.index_to_acts = {i: a for i, a in enumerate(data.act.unique())}
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}
        print(self.acts_to_index)
        print(self.index_to_acts)
        # encoding takes place in SequenceDataset
        return SequenceDataset(
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
        print("decode")
        print(encoded.shape)
        encoded, durations = torch.split(
            encoded, [len(self.acts_to_index) + 2, 1], dim=-1
        )
        print(encoded.shape, durations.shape)
        encoded = torch.argmax(encoded, dim=-1)
        sos_idx = len(self.acts_to_index)
        eos_idx = sos_idx + 1
        print("EOS", eos_idx)
        print("SOS", sos_idx)
        decoded = []

        for pid in range(len(encoded)):
            act_start = 0
            for act_idx, duration in zip(encoded[pid], durations[pid]):
                if int(act_idx) == sos_idx:
                    continue
                if int(act_idx) == eos_idx:
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


class SequenceDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        max_length: int,
        acts_to_index: dict,
        norm_duration: int,
    ):
        """Torch Dataset for sequence data.

        Args:
            data (DataFrame): Population of sequences.
            max_length (int): Max length of sequences.
            acts_to_index (dict): Mapping of activity to index.
        """
        self.encoded = self.encode(
            data, max_length, acts_to_index, norm_duration
        )
        print(self.encoded.shape)
        self.size = len(self.encoded)

    def encode(
        self,
        data: pd.DataFrame,
        max_length: int,
        acts_to_index: dict,
        norm_duration: int,
    ) -> Tensor:
        data = data.copy()
        data.act = data.act.map(acts_to_index)
        data.duration = data.duration / norm_duration
        persons = data.pid.nunique()
        encoding_width = (
            len(acts_to_index) + 3
        )  # one-hot act encoding plus SOS & EOS & duration
        encoded = np.zeros(
            (persons, max_length, encoding_width), dtype=np.float32
        )
        for pid, (_, trace) in enumerate(data.groupby("pid")):
            encoded[pid] = encode_sequence(
                acts=list(trace.act),
                durations=list(trace.duration),
                max_length=max_length,
                encoding_width=encoding_width,
            )
            # [N, L, W]

        return torch.from_numpy(encoded)

    def shape(self):
        return self.encoded[0].shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.encoded[idx]


def encode_sequence(
    acts: list[int], durations: list[int], max_length: int, encoding_width: int
) -> np.ndarray:
    """Create sequence encoding from ranges.

    Args:
        acts (Iterable[int]): _description_
        durations (Iterable[int]): _description_
        max_length (int): _description_
        encoding_width (dict): _description_

    Returns:
        np.array: _description_
    """
    encoding = np.zeros((max_length, encoding_width), dtype=np.int8)
    # SOS
    encoding[0][-3] = 1
    for i in range(1, max_length):
        if i < len(acts):
            act = acts[i]
            duration = durations[i]
            encoding[i][act] = 1
            encoding[i][-1] = duration
        else:
            encoding[i][-2] = 1
    return encoding
