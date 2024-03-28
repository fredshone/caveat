import torch

from caveat.encoders import BaseDataset, PaddedDatatset


def test_base_encoded():
    encoded = BaseDataset(
        schedules=torch.rand((3, 12)),
        masks=torch.ones((3, 12)),
        activity_encodings=4,
        activity_weights=torch.ones(4),
        augment=None,
        conditionals=torch.ones((3, 3)),
    )
    for i in range(len(encoded)):
        (left, left_mask), (right, right_mask), attributes = encoded[i]
        assert left.shape == (12,)
        assert left_mask.shape == (12,)
        assert right.shape == (12,)
        assert right_mask.shape == (12,)
        assert attributes.shape == (3,)


def test_base_encoded_padded():
    encoded = PaddedDatatset(
        schedules=torch.rand((3, 12)),
        masks=torch.ones((3, 13)),
        activity_encodings=4,
        activity_weights=torch.ones(4),
        augment=None,
        conditionals=torch.ones((3, 3)),
    )
    for i in range(len(encoded)):
        (left, left_mask), (right, right_mask), attributes = encoded[i]
        assert left.shape == (13,)
        assert left_mask.shape == (13,)
        assert right.shape == (13,)
        assert right_mask.shape == (13,)
        assert attributes.shape == (3,)
