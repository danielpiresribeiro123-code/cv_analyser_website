"""
CV detection module using YOLO.

This module provides:
• A reusable CVFilter class
• Support for cached YOLO model injection
• Robust image handling (path, numpy array)
• Clean production-ready API
"""

from pathlib import Path
from typing import Union, Optional

import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore

# Centralized model loader
from services.paths_models_and_data import get_filter_yolo_model


# =========================================================
# CV FILTER CLASS
# =========================================================

class CVFilter:
    """
    Detect whether an image corresponds to a CV document.

    Uses a YOLO model trained to detect dense text regions
    typical of CV layouts.
    """

    def __init__(self, model: Optional[YOLO] = None) -> None:
        """
        Initialize CVFilter.

        Args:
            model:
                Preloaded YOLO model instance.
                If None → loads cached model via central loader.
        """
        self.model: YOLO = model if model is not None else get_filter_yolo_model()

    # -----------------------------------------------------
    # Core detection
    # -----------------------------------------------------

    def _run_inference(
        self,
        image: np.ndarray,
        confidence_threshold: float,
    ):
        """Run YOLO inference safely."""
        return self.model(image, conf=confidence_threshold)

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def is_cv(
        self,
        image: Union[str, Path, np.ndarray],
        confidence_threshold: float = 0.3,
        min_detections: int = 5,
    ) -> bool:
        """
        Determine whether an image is likely a CV.

        Args:
            image:
                Can be:
                • file path
                • numpy BGR image
            confidence_threshold:
                YOLO confidence threshold.
            min_detections:
                Minimum number of detected text boxes to classify as CV.

        Returns:
            True if image resembles a CV layout.
        """

        # -------------------------------------------------
        # Load image if path provided
        # -------------------------------------------------
        if isinstance(image, (str, Path)):
            image_path = Path(image)

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")

        elif isinstance(image, np.ndarray):
            img = image

        else:
            raise TypeError("image must be a path or numpy.ndarray")

        # -------------------------------------------------
        # Run YOLO inference
        # -------------------------------------------------
        results = self._run_inference(img, confidence_threshold)

        if not results:
            return False

        boxes = getattr(results[0], "boxes", None)

        num_detections = len(boxes) if boxes is not None else 0

        # -------------------------------------------------
        # CV heuristic:
        # Dense text regions → many detections
        # -------------------------------------------------
        return num_detections >= min_detections


# =========================================================
# FUNCTIONAL HELPERS (LIGHTWEIGHT API)
# =========================================================

def identify_cv(
    image: Union[str, Path, np.ndarray],
    model: Optional[YOLO] = None,
    confidence_threshold: float = 0.3,
    min_detections: int = 5,
) -> bool:
    """
    Standalone helper to check if image is a CV.

    Useful for quick scripts without instantiating CVFilter.
    """
    return CVFilter(model=model).is_cv(
        image=image,
        confidence_threshold=confidence_threshold,
        min_detections=min_detections,
    )


def is_cv_image(
    image: Union[str, Path, np.ndarray],
    model: Optional[YOLO] = None,
    confidence_threshold: float = 0.3,
    min_detections: int = 5,
) -> bool:
    """
    Alias for identify_cv (semantic readability).
    """
    return identify_cv(
        image=image,
        model=model,
        confidence_threshold=confidence_threshold,
        min_detections=min_detections,
    )
