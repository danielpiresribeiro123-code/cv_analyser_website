from pathlib import Path
import shutil
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
            print(f"Failed to delete {item}: {e}")


def clear_yolo_runs(project_root: Path) -> None:
    """
    Remove YOLO-generated 'runs' directories inside the project.
    Only deletes known Ultralytics run folders for safety.
    """

    if not project_root.exists():
        return

    for path in project_root.rglob("runs"):
        if path.is_dir() and any(
            (path / sub).exists() for sub in ("detect", "train", "predict")
        ):
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Failed to delete {path}: {e}")



# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def clear_uploads(uploads_dir: Path) -> None:
    """Clear uploaded CV files."""
    _safe_clear_directory(uploads_dir)


def clear_outputs(outputs_dir: Path) -> None:
    """Clear pipeline outputs and CSV results."""
    _safe_clear_directory(outputs_dir)


def clear_all_storage(uploads_dir: Path,outputs_dir: Path,project_root: Path) -> None:
    """
    Clear uploads, outputs, and YOLO 'runs' folders.
    """
    # clear_uploads(uploads_dir)
    clear_outputs(outputs_dir)
    clear_yolo_runs(project_root)





def clean_df_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make DataFrame JSON-serializable:
    - replace NaN
    - replace ±inf
    """

    df = df.replace([np.inf, -np.inf], None)
    df = df.astype(object).where(pd.notnull(df), None) #type: ignore

    return df


