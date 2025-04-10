import os
import pandas as pd
from typing import Dict, Tuple
from tqdm import tqdm

class DataLoader:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._load_csvs()

    def _load_csvs(self):
        for filename in tqdm(os.listdir(self.data_dir)):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.data_dir, filename)
                df_name = os.path.splitext(filename)[0]
                self._dataframes[df_name] = pd.read_csv(filepath)

    def get(self, name: str) -> pd.DataFrame:
        return self._dataframes.get(name)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.get(name)

    def keys(self):
        return self._dataframes.keys()

    def items(self):
        return self._dataframes.items()

    def values(self):
        return self._dataframes.values()

    def split_by_gender(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        mens = {}
        womens = {}
        general = {}

        for name, df in self._dataframes.items():
            if name.startswith("M") and len(name) > 1 and name[1].isupper():
                mens[name[1:]] = df  # Strip the "M"
            elif name.startswith("W") and len(name) > 1 and name[1].isupper():
                womens[name[1:]] = df  # Strip the "W"
            else:
                general[name] = df

        return mens, womens, general