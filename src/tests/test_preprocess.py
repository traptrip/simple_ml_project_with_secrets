import os
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from preprocess import preprocess, split_data


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.tiny_data_dir = Path(__file__).parent / "../../tests/tiny_dataset"
        self.tiny_data = pd.read_csv(self.tiny_data_dir / "train.csv")

    def test_split_data(self):
        processed_data = split_data(self.tiny_data)
        assert (
            "stage" in processed_data.columns
            and processed_data.loc[processed_data.stage == "val"].shape[0] > 0
        )

    def test_save_splitted_data(self):
        self.assertEqual(
            preprocess(self.tiny_data_dir),
            True,
        )


if __name__ == "__main__":
    unittest.main()
