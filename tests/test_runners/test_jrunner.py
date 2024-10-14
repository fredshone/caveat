import torch

from caveat.jrunners import repack_labels


def test_repack_single_batch_labels():
    i = ([torch.randn((128, 2)), torch.randn((128, 3))],)
    o = repack_labels(i)
    assert len(o) == 2
    assert o[0].shape == (128, 2)
    assert o[1].shape == (128, 3)


def test_repack_n_batch_labels():
    i = ([torch.randn((128, 2)), torch.randn((128, 3))] for _ in range(3))
    o = repack_labels(i)
    assert len(o) == 2
    assert o[0].shape == (128 * 3, 2)
    assert o[1].shape == (128 * 3, 3)
