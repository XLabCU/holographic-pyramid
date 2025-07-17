"""
Image processing module for hologram pyramid display.
Handles dynamic, aspect-ratio-correct scaling and layout.
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class View(Enum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"

@dataclass
class ProcessingConfig:
    """Configuration for the image processor."""
    # This controls the height of the final generated image.
    # Higher values produce a higher resolution output.
    base_height: int = 1080
    interpolation: int = cv2.INTER_AREA

class ImageProcessor:
    """
    Handles the creation of a dynamically sized, aspect-ratio-correct 
    cross layout for a Pepper's Ghost pyramid.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        # these angles are arranged for a pyramid whose 'top' sits on the display
        # if you want the opposite (so that images are projected against the inside of the pyramid)
        # you'd adjust these 180.
        self.rotations = {
            View.FRONT: 0,
            View.BACK: 180,
            View.LEFT: 270,
            View.RIGHT: 90,
        }
        logger.info(f"ImageProcessor initialized with base height: {self.config.base_height}")

    # In core/processor.py

    def create_pyramid_layout(self, source_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Creates the complete pyramid layout using a robust 3x3 grid system
        to prevent any view overlap.
        """
        if source_image is None or source_image.size == 0:
            return None

        # --- Stage 1: Establish a single, consistent scale factor ---
        # The scale is determined by the desired final height of the side views,
        # which after rotation, corresponds to the source image's width.
        side_view_target_h = self.config.base_height // 2
        src_h, src_w = source_image.shape[:2]
        if src_w == 0:
            return None
        
        scale = side_view_target_h / src_w

        # --- Stage 2: Process all four views with the same scale factor ---
        # This ensures the "thickness" of the cross's arms is consistent.
        processed_left = self._process_view(source_image, scale, self.rotations[View.LEFT])
        processed_right = self._process_view(source_image, scale, self.rotations[View.RIGHT])
        processed_front = self._process_view(source_image, scale, self.rotations[View.FRONT])
        processed_back = self._process_view(source_image, scale, self.rotations[View.BACK])

        if processed_left is None or processed_front is None:
            return None

        # --- Stage 3: Define the dimensions of the 3x3 grid cells ---
        # The "thickness" of the arms (the dimension that meets in the middle).
        # This is the height of the side views and the width of the top/bottom views.
        thickness = processed_left.shape[0]

        # The length of the horizontal arms (left/right views).
        # This is the width of the side views.
        h_arm_length = processed_left.shape[1]

        # The length of the vertical arms (top/bottom views).
        # This is the height of the top/bottom views.
        v_arm_length = processed_front.shape[0]

        # --- Stage 4: Create the final canvas and paste the views into the grid ---
        final_width = h_arm_length + thickness + h_arm_length
        final_height = v_arm_length + thickness + v_arm_length
        final_layout = np.zeros((final_height, final_width, 3), dtype=np.uint8)

        # Paste FRONT view into the top-center cell
        x = h_arm_length
        y = 0
        self._paste_image(final_layout, processed_front, x, y)

        # Paste BACK view into the bottom-center cell
        x = h_arm_length
        y = v_arm_length + thickness
        self._paste_image(final_layout, processed_back, x, y)

        # Paste LEFT view into the middle-left cell
        x = 0
        y = v_arm_length
        self._paste_image(final_layout, processed_left, x, y)

        # Paste RIGHT view into the middle-right cell
        x = h_arm_length + thickness
        y = v_arm_length
        self._paste_image(final_layout, processed_right, x, y)

        return final_layout

    def _process_view(self, image: np.ndarray, scale: float, angle: int) -> Optional[np.ndarray]:
        """Helper to scale and rotate an image."""
        try:
            resized = self._resize_image(image, scale)
            if resized is None: return None
            return self._rotate_image(resized, angle)
        except Exception as e:
            logger.error(f"Error in _process_view: {e}")
            return None

    def _resize_image(self, image: np.ndarray, scale: float) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0: return None
        interpolation = self.config.interpolation if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _paste_image(self, canvas: np.ndarray, image: np.ndarray, x: int, y: int):
        """Helper to safely paste an image onto a canvas."""
        if image is None: return
        h, w = image.shape[:2]
        canvas[y:y+h, x:x+w] = image