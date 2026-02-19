from pathlib import Path
import shutil
from typing import Iterable
import pandas as pd
import numpy as np

def _safe_clear_directory(path: Path) -> None:
    """
    Remove all files and folders inside a directory,
    but keeps the directory itself.
    """
    if not path.exists():
        return

    for item in path.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"⚠️ Failed to delete {item}: {e}")


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def clear_uploads(uploads_dir: Path) -> None:
    """Clear uploaded CV files."""
    _safe_clear_directory(uploads_dir)


def clear_outputs(outputs_dir: Path) -> None:
    """Clear pipeline outputs and CSV results."""
    _safe_clear_directory(outputs_dir)


def clear_all_storage(uploads_dir: Path, outputs_dir: Path) -> None:
    """Clear both uploads and outputs."""
    clear_uploads(uploads_dir)
    clear_outputs(outputs_dir)





def clean_df_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make DataFrame JSON-serializable:
    - replace NaN
    - replace ±inf
    """

    df = df.replace([np.inf, -np.inf], None)
    df = df.astype(object).where(pd.notnull(df), None) #type: ignore

    return df
