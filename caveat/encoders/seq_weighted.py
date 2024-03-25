from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from caveat.data.augment import SequenceJitter
from caveat.encoders import AttributeEncoder, BaseEncoded, BaseEncoder


class SequenceEncoder(BaseEncoder):
    def __init__(self, max_length: int = 12, duration: int = 1440, **kwargs):
        """Sequence Encoder for sequences of activities. Also supports conditional attributes.

        Args:
            max_length (int, optional): _description_. Defaults to 12.
            duration (int, optional): _description_. Defaults to 1440.
        """
        self.max_length = max_length
        self.duration = duration
        self.jitter = kwargs.get("jitter", 0)
        self.conditionals = kwargs.get("conditionals")
        print(
            f"WeightedSequenceEncoder: {self.max_length=}, {self.duration=}, {self.jitter=}, {self.conditionals=}"
        )
        if self.conditionals:
            self.attribute_encoder = AttributeEncoder(self.conditionals)
        else:
            self.attribute_encoder = None

    def encode(
        self, sequences: pd.DataFrame, attributes: Optional[pd.DataFrame] = None
    ) -> BaseEncoded:
        self.sos = 0
        self.eos = 1
        self.index_to_acts = {
            i + 2: a for i, a in enumerate(sequences.act.unique())
        }
        self.index_to_acts[0] = "<SOS>"
        self.index_to_acts[1] = "<EOS>"
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}
        self.encodings = len(self.index_to_acts)

        # encode attributes
        if attributes is not None:
            if self.conditionals is None:
                raise UserWarning(
                    "Attributes provided but no conditionals configured."
                )
            encoded_attributes = self.attribute_encoder.encode(attributes)
        else:
            encoded_attributes = None

        return SequenceDurationsWeighted(
            data=sequences,
            max_length=self.max_length,
            acts_to_index=self.acts_to_index,
            norm_duration=self.duration,
            jitter=self.jitter,
            attributes=encoded_attributes,
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
                duration = int(duration * self.duration)
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


class SequenceDurationsWeighted(BaseEncoded):
    def __init__(
        self,
        data: pd.DataFrame,
        max_length: int,
        acts_to_index: dict,
        norm_duration: int,
        jitter: float = 0,
        attributes: Optional[Tensor] = None,
        sos: int = 0,
        eos: int = 1,
    ):
        """Torch Dataset for sequence data.

        Args:
            data (DataFrame): Population of sequences.
            max_length (int): Max length of sequences.
            acts_to_index (dict): Mapping of activity to index.
            norm_duration (int): Normal day duration.
            jitter (float, optional): Jitter to apply to sequences. Defaults to 0.
            attributes (Optional[Tensor], optional): Conditional attributes. Defaults to None.
        """
        self.max_length = max_length
        if jitter:
            self.augment = SequenceJitter(jitter)
        else:
            self.augment = None
        self.sos = sos
        self.eos = eos
        self.encodings = len(acts_to_index)
        self.encoded, self.encoding_weights = self._encode_sequences(
            data, max_length, acts_to_index, norm_duration
        )
        self.attributes = attributes

        self.size = len(self.encoded)
        self.weights = None

    def _encode_sequences(
        self,
        data: pd.DataFrame,
        max_length: int,
        acts_to_index: dict,
        norm_duration: int,
    ) -> Tuple[Tensor, Tensor]:
        data = data.copy()
        data.duration = data.duration / norm_duration
        data.act = data.act.map(acts_to_index)

        # calc weightings
        act_weights = (
            data.groupby("act", observed=True).duration.sum().to_dict()
        )
        n = data.pid.nunique()  # sos/eos weight equal to 1 day
        act_weights.update({self.sos: n, self.eos: n})
        act_weights = np.array(
            [act_weights[k] for k in range(len(act_weights))]
        )
        act_weights = 1 / act_weights

        persons = data.pid.nunique()
        encoding_width = 2  # cat act encoding plus duration

        encoded = np.zeros(
            (persons, max_length, encoding_width), dtype=np.float32
        )
        weights = np.zeros((persons, max_length), dtype=np.float32)

        for pid, (_, trace) in enumerate(data.groupby("pid")):
            seq_encoding, seq_weights = encode_sequence(
                acts=list(trace.act),
                durations=list(trace.duration),
                max_length=max_length,
                encoding_width=encoding_width,
                act_weights=act_weights,
                sos=self.sos,
                eos=self.eos,
            )
            encoded[pid] = seq_encoding  # [N, L, W]
            weights[pid] = seq_weights  # [N, L]

        return (torch.from_numpy(encoded), torch.from_numpy(weights))

    def shape(self):
        return self.encoded[0].shape

    def conditionals_shape(self):
        if self.attributes is None:
            return None
        return self.attributes[0].shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self.encoded[idx]
        if self.augment:
            sample = self.augment(sample)
        mask = self.encoding_weights[idx]
        if self.attributes is None:
            return (sample, mask), (sample, mask), None
        return (sample, mask), (sample, mask), self.attributes[idx]


def encode_sequence(
    acts: list[int],
    durations: list[float],
    max_length: int,
    encoding_width: int,
    act_weights: np.ndarray,
    sos: int,
    eos: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequence encoding from ranges.

    Args:
        acts (Iterable[int]): _description_
        durations (Iterable[float]): _description_
        max_length (int): _description_
        encoding_width (dict): _description_
        act_weights (dict): _description_
        sos (int): _description_
        eos (int): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
    """
    encoding = np.zeros((max_length, encoding_width), dtype=np.float32)
    weights = np.zeros((max_length), dtype=np.float32)
    # SOS
    encoding[0][0] = sos
    # mask includes sos
    weights[0] = act_weights[sos]

    for i in range(1, max_length):
        if i < len(acts) + 1:
            encoding[i][0] = acts[i - 1]
            encoding[i][1] = durations[i - 1]
            weights[i] = act_weights[acts[i - 1]]
        elif i < len(acts) + 2:
            encoding[i][0] = eos
            # mask includes first eos
            weights[i] = act_weights[eos]
        else:
            encoding[i][0] = eos
            # act weights are 0 for padding eos

    return encoding, weights
