"""Simple test runner for the CV processing pipeline.

Usage:
    python -m backend.pipeline.pipelines.run_test_process_single_cv <pdf_path> [output_csv]
"""

import sys
from pathlib import Path

from backend.pipeline.pipelines.process_single_cv_v3 import process_single_cv


def main(argv):
    if len(argv) < 2:
        print("Usage: python -m backend.pipeline.pipelines.run_test_process_single_cv <pdf_path> [output_csv]")
        return 1

    pdf_path = Path(argv[1])
    out_csv = Path(argv[2]) if len(argv) > 2 else None
    json_path = Path(argv[3]) if len(argv) > 3 else None
    print("JSON PATH TEST:", json_path)

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return 1

    try:
        df = process_single_cv(pdf_path, save_csv=out_csv, save_json_path =json_path, verbose=True)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return 1

    if df is None or df.empty:
        print("Pipeline finished but no CV pages were detected or extracted.")
        return 2

    print("\nPipeline returned DataFrame with the following columns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
