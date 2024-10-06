import pandas as pd

from caveat.samplers import TargetLabelSampler


def test_target_label_sampler():
    target_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "b", "c"]}
    ).set_index("pid")
    sample_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a_", "b", "c"]}
    )
    sample_schedules = pd.DataFrame(
        {"pid": [1, 2, 3], "schedule": ["A", "B", "C"]}
    ).set_index("pid")
    sampler = TargetLabelSampler(target_labels, ["label_a", "label_b"])
    sampler.sample(sample_labels, sample_schedules)
    sampled_labels, sampled_schedules = sampler.finish()
    expected_labels = pd.DataFrame(
        {"pid": [0, 1], "label_a": [2, 3], "label_b": ["b", "c"]}
    ).set_index("pid")
    print(expected_labels)
    print(sampled_labels)
    pd.testing.assert_frame_equal(expected_labels, sampled_labels)
    expected_schedules = pd.DataFrame(
        {"pid": [0, 1], "schedule": ["B", "C"]}
    ).set_index("pid")
    pd.testing.assert_frame_equal(expected_schedules, sampled_schedules)
