from pathlib import Path

import pandas as pd


def load_and_validate(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    if data.empty:
        raise UserWarning(f"No data found in {data_path}.")
    validate(data)
    return data.sort_values(by=["pid", "start"])


def load_and_validate_attributes(
    data_path: Path, sequences: pd.DataFrame
) -> pd.DataFrame:
    attributes = pd.read_csv(data_path)
    if attributes.empty:
        raise UserWarning(f"No attributes found in {data_path}.")
    seq_index = sequences.pid
    attr_index = attributes.pid
    if not set(seq_index) == set(attr_index):
        raise UserWarning("Sequence and attributes pid do not match")
    if not seq_index.dtype == attr_index.dtype:
        raise UserWarning("Sequence and attributes pid datatypes do not match")
    return attributes.sort_values(by="pid")


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
