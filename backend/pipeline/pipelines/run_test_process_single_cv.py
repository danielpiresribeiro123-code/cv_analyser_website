"""
Simple test runner for the CV processing pipeline.

Usage:
    python -m backend.pipeline.pipelines.run_test_process_single_cv <pdf_path> [output_folder]

Optional flags (edit inside main if needed):
    - crop_photo: save detected profile photos
    - crop_skill_table: save detected skill tables
"""

# =========================================================
# Standard library imports
# =========================================================
import sys
from pathlib import Path

# =========================================================
# Local imports
# =========================================================
from backend.pipeline.pipelines.process_single_cv_v3 import process_single_cv #type: ignore


def main(argv) -> int:
    """
    CLI entry point for testing the end-to-end CV pipeline.

    Steps:
        1. Parse CLI arguments
        2. Validate PDF path
        3. Execute pipeline
        4. Print summary of extracted results

    Returns
    -------
    int
        Exit code:
            0 → success
            1 → execution error / invalid input
            2 → pipeline ran but no CV detected
    """

    # -----------------------------------------------------
    # [1] Argument parsing
    # -----------------------------------------------------
    if len(argv) < 2:
        print(
            "Usage: python -m backend.pipeline.pipelines.run_test_process_single_cv "
            "<pdf_path> [output_folder]"
        )
        return 1

    pdf_path = Path(argv[1])
    save_path = Path(argv[2]) if len(argv) > 2 else None

    # -----------------------------------------------------
    # [2] Validate input
    # -----------------------------------------------------
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return 1

    # -----------------------------------------------------
    # [3] Configure optional behavior
    # (edit here if you want different defaults)
    # -----------------------------------------------------
    crop_photo = True
    crop_skill_table = False

    # -----------------------------------------------------
    # [4] Run pipeline
    # -----------------------------------------------------
    try:
        df = process_single_cv(
            pdf_input=pdf_path,
            save_path=save_path,
            crop_photo=crop_photo,
            crop_skill_table=crop_skill_table,
            verbose=True,
        )
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return 1

    # -----------------------------------------------------
    # [5] Handle empty results
    # -----------------------------------------------------
    if df is None or df.empty:
        print("Pipeline finished but no CV pages were detected or extracted.")
        return 2

    # -----------------------------------------------------
    # [6] Display results summary
    # -----------------------------------------------------
    print("\nPipeline returned DataFrame with the following columns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head().to_string())

    print("\n✓ Test completed successfully.")
    return 0


# =========================================================
# CLI execution
# =========================================================
if __name__ == "__main__":
    sys.exit(main(sys.argv))
