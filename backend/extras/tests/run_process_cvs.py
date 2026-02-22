from services.process_cvs_pipeline import process_folder_cvs
from services.paths_models_and_data import BACKEND_ROOT

if __name__ == "__main__":
    df = process_folder_cvs(
        input_folder=BACKEND_ROOT / "uploads",
        output_folder=BACKEND_ROOT / "outputs",
        scan_subfolders=True,
        verbose=True,
        clear_output_folder=True,
        crop_text_block=True,
        crop_photo=True,
        crop_skill_table=True,
        save_ocr_txt=True,
        analyse_individual_pages=False,
        cv_confidence_threshold=0.3,
        yolo_confidence=0.25,
    )

    print("\nFinal DataFrame shape:", df.shape)
