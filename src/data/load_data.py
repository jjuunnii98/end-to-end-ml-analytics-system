from pathlib import Path
import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(path)