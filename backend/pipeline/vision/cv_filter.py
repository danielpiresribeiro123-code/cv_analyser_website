"""
Module for filtering and identifying CV (Curriculum Vitae) images.
Uses a trained YOLO model to detect text patterns and identify CV documents.
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np
from ultralytics import YOLO #type: ignore

# =========================================================
# Project root resolution (no path_loader required)
#
# Current file path:
# backend\pipeline\pipelines\process_single_cv.py
#
# parents[0] -> pipelines
# parents[1] -> pipeline
# parents[2] -> backend
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILTER_MODEL_PATH = PROJECT_ROOT / "models" / "yolo"/"yolo11n-text.pt"

class CVFilter:
    """Class for identifying whether an image is a CV using YOLO text detection."""
    
    def __init__(self, model_path: Union[str, Path] | None= None):
        """
        Initialize the CV Filter with a trained YOLO model.
        
        Args:
            model_path: Path to the trained YOLO model. If None, uses default path.
        """
        if model_path is None:
            # Default path to the trained YOLO model
            model_path = FILTER_MODEL_PATH
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load the trained YOLO model
        self.model = YOLO(str(self.model_path))
    
    def is_cv(
        self, 
        image_path: Union[str, Path],
        confidence_threshold: float = 0.3,
        min_detections: int = 5
    ) -> bool:
        """
        Determine if an image is a CV based on text detection.
        
        Args:
            image_path: Path to the image file to analyze.
            confidence_threshold: Minimum confidence score for detections (0-1).
            min_detections: Minimum number of text detections to consider it a CV.
        
        Returns:
            Boolean indicating if image is likely a CV.
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Run inference
        results = self.model(image, conf=confidence_threshold)
        
        # Extract detections
        detections = results[0]
        
        # Count detections
        num_detections = 0
        if detections.boxes is not None:
            num_detections = len(detections.boxes)
        
        # A CV typically has multiple text regions detected
        return num_detections >= min_detections
    
    def is_cv_simple(
        self,
        image_path: Union[str, Path],
        confidence_threshold: float = 0.3,
        min_detections: int = 5
    ) -> bool:
        """
        Check if image is a CV. Alias for is_cv() for consistency.
        
        Args:
            image_path: Path to the image file to analyze.
            confidence_threshold: Minimum confidence score for detections (0-1).
            min_detections: Minimum number of text detections to consider it a CV.
        
        Returns:
            Boolean indicating if image is likely a CV.
        """
        return self.is_cv(image_path, confidence_threshold, min_detections)


def identify_cv(
    image_path: Union[str, Path],
    model_path: Union[str, Path] | None= None,
    confidence_threshold: float = 0.3,
    min_detections: int = 5
) -> bool:
    """
    Identify whether an image is a CV.
    
    Standalone function that creates a CVFilter instance and analyzes the image.
    
    Args:
        image_path: Path to the image file to analyze.
        model_path: Path to the trained YOLO model. If None, uses default path.
        confidence_threshold: Minimum confidence score for detections (0-1).
        min_detections: Minimum number of text detections to consider it a CV.
    
    Returns:
        Boolean indicating if image is likely a CV.
    
    Example:
        >>> if identify_cv('path/to/image.jpg'):
        ...     print("This is a CV")
    """
    cv_filter = CVFilter(model_path=model_path)
    return cv_filter.is_cv(image_path, confidence_threshold, min_detections)


def is_cv_image(
    image_path: Union[str, Path],
    model_path: Union[str, Path] | None = None,
    confidence_threshold: float = 0.3,
    min_detections: int = 5
) -> bool:
    """
    Simple boolean check if image is a CV.
    """
    return identify_cv(image_path, model_path, confidence_threshold, min_detections)
