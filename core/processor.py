"""
Image processing module for hologram pyramid display.
Handles rotation, scaling, and quadrant arrangement for the Pepper's Ghost illusion.
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class QuadrantPosition(Enum):
    """Enumeration for pyramid quadrant positions."""
    TOP_LEFT = "top_left"      # Back view
    TOP_RIGHT = "top_right"    # Right view
    BOTTOM_LEFT = "bottom_left"   # Left view
    BOTTOM_RIGHT = "bottom_right" # Front view

@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""
    content_scale: float = 0.9          # Scale content to leave border
    interpolation: int = cv2.INTER_AREA # Interpolation method for resizing
    background_color: Tuple[int, int, int] = (0, 0, 0)  # Background color (BGR)
    enable_antialiasing: bool = True     # Enable antialiasing for rotations
    quality_mode: str = "balanced"       # "fast", "balanced", "high_quality"

class ImageProcessor:
    """Handles image processing operations for pyramid display with improved flexibility."""

    def __init__(self, output_width: int = 1920, output_height: int = 1080, config: Optional[ProcessingConfig] = None):
        self.output_width = output_width
        self.output_height = output_height
        self.quadrant_width = output_width // 2
        self.quadrant_height = output_height // 2
        
        # Processing configuration
        self.config = config or ProcessingConfig()
        
        # Rotation angles for each view of the pyramid
        self.rotations = {
            QuadrantPosition.TOP_LEFT: 180,      # Back view
            QuadrantPosition.TOP_RIGHT: 270,     # Right view  
            QuadrantPosition.BOTTOM_LEFT: 90,    # Left view
            QuadrantPosition.BOTTOM_RIGHT: 0,    # Front view
        }
        
        # Performance optimization: pre-calculate common values
        self._update_derived_values()
        
        logger.info(f"ImageProcessor initialized: {output_width}x{output_height}, quadrants: {self.quadrant_width}x{self.quadrant_height}")

    def _update_derived_values(self):
        """Update derived values when configuration changes."""
        self.effective_quad_width = int(self.quadrant_width * self.config.content_scale)
        self.effective_quad_height = int(self.quadrant_height * self.config.content_scale)

    def set_output_resolution(self, width: int, height: int):
        """Update output resolution and recalculate quadrant dimensions."""
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
            
        self.output_width = width
        self.output_height = height
        self.quadrant_width = width // 2
        self.quadrant_height = height // 2
        self._update_derived_values()
        
        logger.info(f"Resolution updated to {width}x{height}")

    def set_config(self, config: ProcessingConfig):
        """Update processing configuration."""
        self.config = config
        self._update_derived_values()
        logger.info("Processing configuration updated")

    def create_pyramid_layout(self, source_image: np.ndarray) -> np.ndarray:
        """
        Create the complete four-quadrant pyramid layout from a source image.
        
        Args:
            source_image: Input image to process
            
        Returns:
            Processed image with four quadrant views arranged for pyramid display
        """
        if source_image is None or source_image.size == 0:
            logger.warning("Invalid source image provided")
            return self._create_empty_frame()

        try:
            # Validate input image
            if len(source_image.shape) != 3 or source_image.shape[2] != 3:
                logger.error(f"Invalid image shape: {source_image.shape}. Expected (H, W, 3)")
                return self._create_empty_frame()

            # Process each quadrant view
            quadrants = {}
            for position in QuadrantPosition:
                try:
                    quadrants[position] = self._process_view(source_image, self.rotations[position])
                except Exception as e:
                    logger.error(f"Error processing quadrant {position}: {e}")
                    quadrants[position] = self._create_empty_quadrant()

            # Arrange quadrants into final layout
            return self._arrange_quadrants(quadrants)
            
        except Exception as e:
            logger.error(f"Error creating pyramid layout: {e}")
            return self._create_empty_frame()

    def _process_view(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        Process a source image for a single view of the pyramid.
        
        Args:
            image: Source image
            angle: Rotation angle in degrees
            
        Returns:
            Processed quadrant image
        """
        try:
            # 1. Calculate optimal scale considering rotation
            scale = self._calculate_optimal_scale(image, angle)
            
            if scale <= 0:
                logger.warning(f"Invalid scale calculated: {scale}")
                return self._create_empty_quadrant()

            # 2. Resize image
            resized = self._resize_image(image, scale)
            if resized is None:
                return self._create_empty_quadrant()

            # 3. Rotate image
            rotated = self._rotate_image(resized, angle)
            if rotated is None:
                return self._create_empty_quadrant()

            # 4. Center in quadrant
            return self._center_in_quadrant(rotated)
            
        except Exception as e:
            logger.error(f"Error processing view with angle {angle}: {e}")
            return self._create_empty_quadrant()

    def _calculate_optimal_scale(self, image: np.ndarray, angle: int) -> float:
        """Calculate optimal scale factor considering rotation."""
        h, w = image.shape[:2]
        
        if w == 0 or h == 0:
            return 0.0
        
        # Account for dimension swap during 90/270 degree rotations
        if angle in [90, 270]:
            target_w, target_h = self.effective_quad_height, self.effective_quad_width
        else:
            target_w, target_h = self.effective_quad_width, self.effective_quad_height
        
        return min(target_w / w, target_h / h)

    def _resize_image(self, image: np.ndarray, scale: float) -> Optional[np.ndarray]:
        """Resize image with the given scale factor."""
        try:
            h, w = image.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            
            if new_w <= 0 or new_h <= 0:
                return None
                
            # Choose interpolation method based on scale
            if scale < 1.0:
                interpolation = cv2.INTER_AREA  # Best for downscaling
            else:
                interpolation = self.config.interpolation
                
            return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return None

    def _rotate_image(self, image: np.ndarray, angle: int) -> Optional[np.ndarray]:
        """Rotate image by the specified angle."""
        try:
            if angle == 0:
                return image
            elif angle == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                # For arbitrary angles, use rotation matrix
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Calculate new dimensions
                cos_angle = abs(matrix[0, 0])
                sin_angle = abs(matrix[0, 1])
                new_w = int((h * sin_angle) + (w * cos_angle))
                new_h = int((h * cos_angle) + (w * sin_angle))
                
                # Adjust translation
                matrix[0, 2] += (new_w / 2) - center[0]
                matrix[1, 2] += (new_h / 2) - center[1]
                
                return cv2.warpAffine(image, matrix, (new_w, new_h), 
                                    flags=cv2.INTER_LINEAR if self.config.enable_antialiasing else cv2.INTER_NEAREST,
                                    borderValue=self.config.background_color)
                
        except Exception as e:
            logger.error(f"Error rotating image by {angle} degrees: {e}")
            return None

    def _center_in_quadrant(self, image: np.ndarray) -> np.ndarray:
        """Center the processed image in a quadrant-sized canvas."""
        try:
            final_h, final_w = image.shape[:2]
            
            # Create quadrant canvas
            quadrant = np.full((self.quadrant_height, self.quadrant_width, 3), 
                              self.config.background_color, dtype=np.uint8)
            
            # Calculate centering offsets
            y_offset = max(0, (self.quadrant_height - final_h) // 2)
            x_offset = max(0, (self.quadrant_width - final_w) // 2)
            
            # Ensure we don't exceed quadrant boundaries
            end_y = min(y_offset + final_h, self.quadrant_height)
            end_x = min(x_offset + final_w, self.quadrant_width)
            
            # Calculate actual region to copy (in case image is larger than quadrant)
            copy_h = end_y - y_offset
            copy_w = end_x - x_offset
            
            if copy_h > 0 and copy_w > 0:
                quadrant[y_offset:end_y, x_offset:end_x] = image[:copy_h, :copy_w]
            
            return quadrant
            
        except Exception as e:
            logger.error(f"Error centering image in quadrant: {e}")
            return self._create_empty_quadrant()

    def _arrange_quadrants(self, quadrants: Dict[QuadrantPosition, np.ndarray]) -> np.ndarray:
        """Arrange the four quadrants into the final layout."""
        try:
            # Arrange into the final layout:
            # Top-Left: Back, Top-Right: Right
            # Bottom-Left: Left, Bottom-Right: Front
            top_row = np.hstack([
                quadrants[QuadrantPosition.TOP_LEFT], 
                quadrants[QuadrantPosition.TOP_RIGHT]
            ])
            bottom_row = np.hstack([
                quadrants[QuadrantPosition.BOTTOM_LEFT], 
                quadrants[QuadrantPosition.BOTTOM_RIGHT]
            ])
            
            return np.vstack([top_row, bottom_row])
            
        except Exception as e:
            logger.error(f"Error arranging quadrants: {e}")
            return self._create_empty_frame()

    def _create_empty_frame(self) -> np.ndarray:
        """Create an empty frame with the output dimensions."""
        return np.full((self.output_height, self.output_width, 3), 
                      self.config.background_color, dtype=np.uint8)

    def _create_empty_quadrant(self) -> np.ndarray:
        """Create an empty quadrant."""
        return np.full((self.quadrant_height, self.quadrant_width, 3), 
                      self.config.background_color, dtype=np.uint8)

    def get_quadrant_info(self) -> Dict[str, any]:
        """Get information about current quadrant configuration."""
        return {
            'output_resolution': (self.output_width, self.output_height),
            'quadrant_size': (self.quadrant_width, self.quadrant_height),
            'effective_size': (self.effective_quad_width, self.effective_quad_height),
            'content_scale': self.config.content_scale,
            'rotations': {pos.value: angle for pos, angle in self.rotations.items()}
        }

    def set_content_scale(self, scale: float):
        """Update content scale factor."""
        if 0.1 <= scale <= 1.0:
            self.config.content_scale = scale
            self._update_derived_values()
            logger.info(f"Content scale updated to {scale}")
        else:
            logger.warning(f"Invalid content scale: {scale}. Must be between 0.1 and 1.0")

    def save_debug_frame(self, frame: np.ndarray, filename: str = "debug_frame.png"):
        """Save a frame for debugging purposes."""
        try:
            cv2.imwrite(filename, frame)
            logger.info(f"Debug frame saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving debug frame: {e}")

    def create_test_pattern(self) -> np.ndarray:
        """Create a test pattern for calibration and debugging."""
        try:
            # Create a test image with various elements
            test_image = np.zeros((400, 600, 3), dtype=np.uint8)
            
            # Add colored rectangles
            test_image[50:150, 50:150] = [255, 0, 0]    # Red
            test_image[50:150, 200:300] = [0, 255, 0]   # Green  
            test_image[250:350, 50:150] = [0, 0, 255]   # Blue
            test_image[250:350, 200:300] = [255, 255, 0] # Yellow
            
            # Add circle
            cv2.circle(test_image, (450, 200), 50, (255, 255, 255), -1)
            
            # Add text
            cv2.putText(test_image, "TEST", (400, 350), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add grid lines for alignment
            for i in range(0, 600, 50):
                cv2.line(test_image, (i, 0), (i, 400), (64, 64, 64), 1)
            for i in range(0, 400, 50):
                cv2.line(test_image, (0, i), (600, i), (64, 64, 64), 1)
            
            return self.create_pyramid_layout(test_image)
            
        except Exception as e:
            logger.error(f"Error creating test pattern: {e}")
            return self._create_empty_frame()