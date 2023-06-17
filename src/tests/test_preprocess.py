from pathlib import Path

import pytest
import pandas as pd

from src.preprocess import preprocess, split_data

DEFAULT_TINY_DATA_DIR = Path(__file__).parent / "../../tests/tiny_dataset"


@pytest.fixture()
def tyny_dataset():
    return pd.read_csv(DEFAULT_TINY_DATA_DIR / "train.csv")


def test_split_data(tyny_dataset):
    processed_data = split_data(tyny_dataset)
    assert (
        "stage" in processed_data.columns
        and processed_data.loc[processed_data.stage == "val"].shape[0] > 0
    )


def test_save_splitted_data():
    assert preprocess(DEFAULT_TINY_DATA_DIR)
