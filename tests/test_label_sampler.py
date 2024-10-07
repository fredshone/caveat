import pandas as pd

from caveat.samplers import TargetLabelSampler


def test_target_label_first_sample():
    target_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "b", "c"]}
    )

    sample_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "X", "X"]}
    )
    sample_schedules = pd.DataFrame(
        {
            "pid": [1, 2, 2, 3, 3, 3],
            "act": ["A", "B", "B", "C", "C", "C"],
            "duration": [3, 1, 2, 1, 1, 1],
        }
    )

    expected_labels = pd.DataFrame(
        {"pid": [0], "label_a": [1], "label_b": ["a"]}
    )
    expected_schedules = pd.DataFrame(
        {"pid": [0], "act": ["A"], "duration": [3]}
    )

    sampler = TargetLabelSampler(target_labels, ["label_a", "label_b"])
    sampler.sample(sample_labels, sample_schedules)
    sampled_labels, sampled_schedules = sampler.finish()

    pd.testing.assert_frame_equal(
        expected_labels, sampled_labels, check_index_type=False
    )
    pd.testing.assert_frame_equal(
        expected_schedules, sampled_schedules, check_index_type=False
    )


def test_target_label_mid_sample():
    target_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "b", "c"]}
    )

    sample_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["X", "b", "X"]}
    )
    sample_schedules = pd.DataFrame(
        {
            "pid": [1, 2, 2, 3, 3, 3],
            "act": ["A", "B", "B", "C", "C", "C"],
            "duration": [3, 1, 2, 1, 1, 1],
        }
    )

    expected_labels = pd.DataFrame(
        {"pid": [0], "label_a": [2], "label_b": ["b"]}
    )
    expected_schedules = pd.DataFrame(
        {"pid": [0, 0], "act": ["B", "B"], "duration": [1, 2]}
    )

    sampler = TargetLabelSampler(target_labels, ["label_a", "label_b"])
    sampler.sample(sample_labels, sample_schedules)
    sampled_labels, sampled_schedules = sampler.finish()

    pd.testing.assert_frame_equal(
        expected_labels, sampled_labels, check_index_type=False
    )
    pd.testing.assert_frame_equal(
        expected_schedules, sampled_schedules, check_index_type=False
    )


def test_target_label_end_sample():
    target_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "b", "c"]}
    )

    sample_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["X", "X", "c"]}
    )
    sample_schedules = pd.DataFrame(
        {
            "pid": [1, 2, 2, 3, 3, 3],
            "act": ["A", "B", "B", "C", "C", "C"],
            "duration": [3, 1, 2, 1, 1, 1],
        }
    )

    expected_labels = pd.DataFrame(
        {"pid": [0], "label_a": [3], "label_b": ["c"]}
    )
    expected_schedules = pd.DataFrame(
        {"pid": [0, 0, 0], "act": ["C", "C", "C"], "duration": [1, 1, 1]}
    )

    sampler = TargetLabelSampler(target_labels, ["label_a", "label_b"])
    sampler.sample(sample_labels, sample_schedules)
    sampled_labels, sampled_schedules = sampler.finish()

    pd.testing.assert_frame_equal(
        expected_labels, sampled_labels, check_index_type=False
    )
    pd.testing.assert_frame_equal(
        expected_schedules, sampled_schedules, check_index_type=False
    )


def test_target_label_under_sample():
    target_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "b", "c"]}
    )

    sample_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["X", "b", "c"]}
    )
    sample_schedules = pd.DataFrame(
        {
            "pid": [1, 2, 2, 3, 3, 3],
            "act": ["A", "B", "B", "C", "C", "C"],
            "duration": [3, 1, 2, 1, 1, 1],
        }
    )

    expected_labels = pd.DataFrame(
        {"pid": [0, 1], "label_a": [2, 3], "label_b": ["b", "c"]}
    )
    expected_schedules = pd.DataFrame(
        {
            "pid": [0, 0, 1, 1, 1],
            "act": ["B", "B", "C", "C", "C"],
            "duration": [1, 2, 1, 1, 1],
        }
    )

    sampler = TargetLabelSampler(target_labels, ["label_a", "label_b"])
    sampler.sample(sample_labels, sample_schedules)
    sampled_labels, sampled_schedules = sampler.finish()

    pd.testing.assert_frame_equal(
        expected_labels, sampled_labels, check_index_type=False
    )
    pd.testing.assert_frame_equal(
        expected_schedules, sampled_schedules, check_index_type=False
    )


def test_target_label_over_sample():
    target_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "b", "c"]}
    )

    sample_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 1, 3], "label_b": ["a", "a", "c"]}
    )
    sample_schedules = pd.DataFrame(
        {
            "pid": [1, 2, 2, 3, 3, 3],
            "act": ["A", "B", "B", "C", "C", "C"],
            "duration": [3, 1, 2, 1, 1, 1],
        }
    )

    expected_labels = pd.DataFrame(
        {"pid": [0, 1], "label_a": [1, 3], "label_b": ["a", "c"]}
    )
    expected_schedules = pd.DataFrame(
        {
            "pid": [0, 1, 1, 1],
            "act": ["A", "C", "C", "C"],
            "duration": [3, 1, 1, 1],
        }
    )

    sampler = TargetLabelSampler(target_labels, ["label_a", "label_b"])
    sampler.sample(sample_labels, sample_schedules)
    sampled_labels, sampled_schedules = sampler.finish()

    pd.testing.assert_frame_equal(
        expected_labels, sampled_labels, check_index_type=False
    )
    pd.testing.assert_frame_equal(
        expected_schedules, sampled_schedules, check_index_type=False
    )


def test_target_label_multi_sample():
    target_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "b", "c"]}
    )

    sample_labels_1 = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["X", "b", "c"]}
    )
    sample_labels_2 = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "X", "c"]}
    )
    sample_schedules = pd.DataFrame(
        {
            "pid": [1, 2, 2, 3, 3, 3],
            "act": ["A", "B", "B", "C", "C", "C"],
            "duration": [3, 1, 2, 1, 1, 1],
        }
    )

    expected_labels = pd.DataFrame(
        {"pid": [0, 1, 2], "label_a": [2, 3, 1], "label_b": ["b", "c", "a"]}
    )
    expected_schedules = pd.DataFrame(
        {
            "pid": [0, 0, 1, 1, 1, 2],
            "act": ["B", "B", "C", "C", "C", "A"],
            "duration": [1, 2, 1, 1, 1, 3],
        }
    )

    sampler = TargetLabelSampler(target_labels, ["label_a", "label_b"])
    sampler.sample(sample_labels_1, sample_schedules)
    sampler.sample(sample_labels_2, sample_schedules)
    sampled_labels, sampled_schedules = sampler.finish()

    pd.testing.assert_frame_equal(
        expected_labels, sampled_labels, check_index_type=False
    )
    pd.testing.assert_frame_equal(
        expected_schedules, sampled_schedules, check_index_type=False
    )


def test_target_label_sub_sample():
    target_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "b", "c"]}
    )

    sample_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "X", "X"]}
    )
    sample_schedules = pd.DataFrame(
        {
            "pid": [1, 2, 2, 3, 3, 3],
            "act": ["A", "B", "B", "C", "C", "C"],
            "duration": [3, 1, 2, 1, 1, 1],
        }
    )

    expected_labels = pd.DataFrame(
        {"pid": [0], "label_a": [1], "label_b": ["a"]}
    )
    expected_schedules = pd.DataFrame(
        {"pid": [0], "act": ["A"], "duration": [3]}
    )

    sampler = TargetLabelSampler(target_labels, ["label_b"])
    sampler.sample(sample_labels, sample_schedules)
    sampled_labels, sampled_schedules = sampler.finish()

    pd.testing.assert_frame_equal(
        expected_labels, sampled_labels, check_index_type=False
    )
    pd.testing.assert_frame_equal(
        expected_schedules, sampled_schedules, check_index_type=False
    )


def test_target_label_zero_sample():
    target_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "b", "c"]}
    )

    sample_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["X", "X", "X"]}
    )
    sample_schedules = pd.DataFrame(
        {
            "pid": [1, 2, 2, 3, 3, 3],
            "act": ["A", "B", "B", "C", "C", "C"],
            "duration": [3, 1, 2, 1, 1, 1],
        }
    )

    expected_labels = pd.DataFrame()
    expected_schedules = pd.DataFrame()

    sampler = TargetLabelSampler(target_labels, ["label_b"])
    sampler.sample(sample_labels, sample_schedules)
    sampled_labels, sampled_schedules = sampler.finish()

    pd.testing.assert_frame_equal(
        expected_labels, sampled_labels, check_index_type=False
    )
    pd.testing.assert_frame_equal(
        expected_schedules, sampled_schedules, check_index_type=False
    )


def test_target_nsample_and_report():
    target_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "b", "c"]}
    )

    sample_labels = pd.DataFrame(
        {"pid": [1, 2, 3], "label_a": [1, 2, 3], "label_b": ["a", "X", "X"]}
    )
    sample_schedules = pd.DataFrame(
        {
            "pid": [1, 2, 2, 3, 3, 3],
            "act": ["A", "B", "B", "C", "C", "C"],
            "duration": [3, 1, 2, 1, 1, 1],
        }
    )

    expected_labels = pd.DataFrame(
        {"pid": [0], "label_a": [1], "label_b": ["a"]}
    )
    expected_schedules = pd.DataFrame(
        {"pid": [0], "act": ["A"], "duration": [3]}
    )

    sampler = TargetLabelSampler(target_labels, ["label_b"])
    for _ in range(10):
        sampler.sample(sample_labels, sample_schedules)
        _ = sampler.is_done()
    sampled_labels, sampled_schedules = sampler.finish()
    sampler.print(verbose=True)

    pd.testing.assert_frame_equal(
        expected_labels, sampled_labels, check_index_type=False
    )
    pd.testing.assert_frame_equal(
        expected_schedules, sampled_schedules, check_index_type=False
    )
