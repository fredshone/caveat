from pathlib import Path

import pandas as pd


def load_and_validate(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    if data.empty:
        raise UserWarning(f"No data found in {data_path}.")
    validate(data)
    return data


def validate(data: pd.DataFrame):
    required_cols = {"pid", "act", "start", "end"}
    found = set(data.columns)
    missing = required_cols - found
    if missing:
        raise UserWarning(
            f"""
    Input data is missing required columns.
    Required: {required_cols}.
    Found: {found}.
    Please add missing: {missing}.
    """
        )
    data.act = data.act.astype("category")
    data.start = data.start.astype("int")
    data.end = data.end.astype("int")

    data["duration"] = data.end - data.start
