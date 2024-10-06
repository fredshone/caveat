import pandas as pd


class TargetLabelSampler:

    def __init__(self, target_labels: pd.DataFrame, target_columns: list):
        self.target_columns = target_columns
        self.target_labels = target_labels[target_columns].copy()
        self.sampled_schedules = []
        self.sampled_labels = []

    def sample(self, labels, schedules):
        print(labels.dtypes)
        for j, (_, target_label) in enumerate(
            self.target_labels.copy().iterrows()
        ):
            for i, (_, label) in enumerate(
                labels[self.target_columns].iterrows()
            ):
                if label.equals(target_label):
                    self.sampled_schedules.append(schedules.iloc[i])
                    self.sampled_labels.append(labels.iloc[i])
                    self.target_labels.drop(index=j, inplace=True)

    def finish(self):
        sampled_labels = pd.concat(self.sampled_labels, axis=1).T.drop(
            "pid", axis=1, errors="ignore"
        )
        sampled_labels["pid"] = range(len(sampled_labels))
        sampled_schedules = pd.concat(self.sampled_schedules, axis=1).T.drop(
            "pid", axis=1, errors="ignore"
        )
        sampled_schedules["pid"] = range(len(sampled_schedules))
        return sampled_labels.set_index("pid"), sampled_schedules.set_index(
            "pid"
        )
