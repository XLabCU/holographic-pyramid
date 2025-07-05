"""
Refactored Core window capture functionality for hologram pyramid display.
Handles real-time desktop window capture with cross-platform support.
"""
import platform
import time
import logging
from typing import Optional, Tuple, List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class CaptureError(Exception):
    """Custom exception for capture-related errors."""
    pass

class CaptureMode(Enum):
    """Enumeration of capture modes."""
    WINDOW = "window"
    REGION = "region"
    DEMO = "demo"

@dataclass
class WindowInfo:
    """Data class for window information."""
    title: str
    handle: Any
    size: Tuple[int, int]
    position: Tuple[int, int]
    visible: bool = True

class BaseCaptureBackend(ABC):
    """Abstract base class for capture backends."""
    
    @abstractmethod
    def get_available_windows(self) -> List[WindowInfo]:
        """Get list of available windows."""
        pass
    
    @abstractmethod
    def capture_window_by_handle(self, handle: Any) -> Optional[np.ndarray]:
        """Capture a specific window by handle."""
        pass
    
    @abstractmethod
    def capture_screen_region(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Capture a screen region."""
        pass
    
    @abstractmethod
    def get_window_info_by_handle(self, handle: Any) -> Optional[WindowInfo]:
        """Get window info by handle."""
        pass
    
    @abstractmethod
    def is_window_valid(self, handle: Any) -> bool:
        """Check if window is valid."""
        pass

class MacOSCaptureBackend(BaseCaptureBackend):
    """macOS-specific capture backend using Quartz."""
    
    def __init__(self):
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if macOS dependencies are available."""
        try:
            import Quartz
            self.Quartz = Quartz
            return True
        except ImportError:
            logger.warning("macOS dependencies not available. Install with: pip install pyobjc-framework-Quartz")
            return False
    
    def get_available_windows(self) -> List[WindowInfo]:
        """Get windows on macOS using Quartz."""
        if not self.available:
            return []
            
        try:
            window_list = self.Quartz.CGWindowListCopyWindowInfo(
                self.Quartz.kCGWindowListOptionOnScreenOnly | 
                self.Quartz.kCGWindowListExcludeDesktopElements,
                self.Quartz.kCGNullWindowID
            )
            
            windows = []
            for window in window_list:
                # Filter out small, unnamed, or transparent windows
                if (window.get('kCGWindowLayer') == 0 and 
                    window.get('kCGWindowName') and 
                    window.get('kCGWindowBounds', {}).get('Width', 0) > 100 and 
                    window.get('kCGWindowBounds', {}).get('Height', 0) > 100):
                    
                    bounds = window['kCGWindowBounds']
                    windows.append(WindowInfo(
                        title=f"{window.get('kCGWindowOwnerName', 'Unknown')} - {window.get('kCGWindowName', '')}",
                        handle=window.get('kCGWindowNumber'),
                        size=(int(bounds['Width']), int(bounds['Height'])),
                        position=(int(bounds['X']), int(bounds['Y']))
                    ))
            return windows
            
        except Exception as e:
            logger.error(f"Error getting macOS windows: {e}")
            raise CaptureError(f"Failed to get macOS windows: {e}")
    
    def capture_window_by_handle(self, handle: Any) -> Optional[np.ndarray]:
        """Capture a specific window by its ID on macOS."""
        if not self.available:
            return None
            
        try:
            image_ref = self.Quartz.CGWindowListCreateImage(
                self.Quartz.CGRectNull,
                self.Quartz.kCGWindowListOptionIncludingWindow,
                handle,
                self.Quartz.kCGWindowImageBoundsIgnoreFraming
            )
            
            if image_ref is None:
                return None
                
            return self._cgimage_to_numpy(image_ref)
            
        except Exception as e:
            logger.error(f"Error capturing macOS window: {e}")
            return None
    
    def capture_screen_region(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Capture a screen region on macOS using Quartz."""
        if not self.available:
            return None
            
        try:
            x, y, x2, y2 = region
            width, height = x2 - x, y2 - y
            
            rect = self.Quartz.CGRectMake(x, y, width, height)
            image_ref = self.Quartz.CGWindowListCreateImage(
                rect,
                self.Quartz.kCGWindowListOptionOnScreenOnly,
                self.Quartz.kCGNullWindowID,
                self.Quartz.kCGWindowImageDefault
            )
            
            if image_ref is None:
                return None
                
            return self._cgimage_to_numpy(image_ref)
            
        except Exception as e:
            logger.error(f"Error capturing macOS region: {e}")
            return None
    
    def _cgimage_to_numpy(self, image_ref) -> Optional[np.ndarray]:
        """Convert CGImage to numpy array."""
        try:
            prov = self.Quartz.CGImageGetDataProvider(image_ref)
            data = self.Quartz.CGDataProviderCopyData(prov)
            width = self.Quartz.CGImageGetWidth(image_ref)
            height = self.Quartz.CGImageGetHeight(image_ref)
            stride = self.Quartz.CGImageGetBytesPerRow(image_ref)
            
            pil_image = Image.frombytes("RGBA", (width, height), data, "raw", "BGRA", stride)
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
            
        except Exception as e:
            logger.error(f"Error converting CGImage: {e}")
            return None
    
    def get_window_info_by_handle(self, handle: Any) -> Optional[WindowInfo]:
        """Get window info by handle on macOS."""
        if not self.available:
            return None
            
        try:
            window_list = self.Quartz.CGWindowListCopyWindowInfo(
                self.Quartz.kCGWindowListOptionOnScreenOnly,
                self.Quartz.kCGNullWindowID
            )
            
            for window_info in window_list:
                if window_info.get('kCGWindowNumber') == handle:
                    bounds = window_info['kCGWindowBounds']
                    return WindowInfo(
                        title=f"{window_info.get('kCGWindowOwnerName', 'Unknown')} - {window_info.get('kCGWindowName', '')}",
                        handle=handle,
                        size=(int(bounds['Width']), int(bounds['Height'])),
                        position=(int(bounds['X']), int(bounds['Y']))
                    )
            return None
            
        except Exception as e:
            logger.error(f"Error getting macOS window info: {e}")
            return None
    
    def is_window_valid(self, handle: Any) -> bool:
        """Check if window is valid on macOS."""
        return self.get_window_info_by_handle(handle) is not None

class WindowsLinuxCaptureBackend(BaseCaptureBackend):
    """Windows/Linux capture backend using pygetwindow and PIL."""
    
    def __init__(self):
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if Windows/Linux dependencies are available."""
        try:
            import pygetwindow as gw
            from PIL import ImageGrab
            self.gw = gw
            self.ImageGrab = ImageGrab
            return True
        except ImportError:
            logger.warning("Windows/Linux dependencies not available. Install with: pip install pygetwindow Pillow")
            return False
    
    def get_available_windows(self) -> List[WindowInfo]:
        """Get windows on Windows/Linux using pygetwindow."""
        if not self.available:
            return []
            
        windows = []
        try:
            for window in self.gw.getAllWindows():
                if (window.title and 
                    window.width > 100 and 
                    window.height > 100 and 
                    window.isVisible):
                    
                    windows.append(WindowInfo(
                        title=window.title,
                        handle=window,
                        position=(window.left, window.top),
                        size=(window.width, window.height),
                        visible=window.isVisible
                    ))
        except Exception as e:
            logger.error(f"Error getting Windows/Linux windows: {e}")
            raise CaptureError(f"Failed to get windows: {e}")
        
        return windows
    
    def capture_window_by_handle(self, handle: Any) -> Optional[np.ndarray]:
        """Capture window on Windows/Linux (uses region capture)."""
        if not self.available:
            return None
            
        try:
            window = handle
            if window.isClosed:
                return None
                
            # Use region capture for the window bounds
            region = (window.left, window.top, window.left + window.width, window.top + window.height)
            return self.capture_screen_region(region)
            
        except Exception as e:
            logger.error(f"Error capturing Windows/Linux window: {e}")
            return None
    
    def capture_screen_region(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Capture screen region on Windows/Linux using PIL."""
        if not self.available:
            return None
            
        try:
            screenshot = self.ImageGrab.grab(bbox=region, all_screens=True)
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error capturing screen region: {e}")
            return None
    
    def get_window_info_by_handle(self, handle: Any) -> Optional[WindowInfo]:
        """Get window info by handle on Windows/Linux."""
        if not self.available:
            return None
            
        try:
            window = handle
            if window.isClosed:
                return None
                
            return WindowInfo(
                title=window.title,
                handle=window,
                position=(window.left, window.top),
                size=(window.width, window.height),
                visible=window.isVisible
            )
        except Exception as e:
            logger.error(f"Error getting Windows/Linux window info: {e}")
            return None
    
    def is_window_valid(self, handle: Any) -> bool:
        """Check if window is valid on Windows/Linux."""
        try:
            return not handle.isClosed
        except:
            return False

class WindowCapture:
    """Main window capture class with cross-platform support."""
    
    def __init__(self):
        self.backend = self._create_backend()
        self.target_window_handle: Optional[Any] = None
        self.capture_region: Optional[Tuple[int, int, int, int]] = None
        self.capture_mode = CaptureMode.WINDOW
        
    def _create_backend(self) -> BaseCaptureBackend:
        """Create appropriate backend for the current platform."""
        system = platform.system()
        
        if system == "Darwin":
            return MacOSCaptureBackend()
        else:
            return WindowsLinuxCaptureBackend()
    
    def get_available_windows(self) -> List[Dict]:
        """Get available windows (maintains backward compatibility)."""
        try:
            windows = self.backend.get_available_windows()
            # Convert to dict format for backward compatibility
            return [
                {
                    'title': w.title,
                    'handle': w.handle,
                    'size': w.size,
                    'position': w.position,
                    'visible': w.visible
                }
                for w in windows
            ]
        except Exception as e:
            logger.error(f"Error getting available windows: {e}")
            return []
    
    def set_target_window(self, window_title: str) -> bool:
        """Set target window by title."""
        try:
            windows = self.backend.get_available_windows()
            for window in windows:
                if window_title.lower() in window.title.lower():
                    return self.set_target_window_by_handle(window.handle)
            return False
        except Exception as e:
            logger.error(f"Error setting target window: {e}")
            return False
    
    def set_target_window_by_handle(self, window_handle: Any) -> bool:
        """Set target window by handle."""
        try:
            self.target_window_handle = window_handle
            self.capture_mode = CaptureMode.WINDOW
            
            # Update capture region based on window
            info = self.get_window_info()
            if info:
                pos = info['position']
                size = info['size']
                self.capture_region = (pos[0], pos[1], pos[0] + size[0], pos[1] + size[1])
                return True
            return False
        except Exception as e:
            logger.error(f"Error setting target window by handle: {e}")
            return False
    
    def set_capture_region(self, x: int, y: int, width: int, height: int):
        """Set capture region."""
        self.target_window_handle = None
        self.capture_region = (x, y, x + width, y + height)
        self.capture_mode = CaptureMode.REGION
    
    def capture_window(self) -> Optional[np.ndarray]:
        """Capture the target window."""
        if not self.is_window_valid():
            return None
            
        try:
            if platform.system() == "Darwin":
                return self.backend.capture_window_by_handle(self.target_window_handle)
            else:
                # For Windows/Linux, update region first
                info = self.get_window_info()
                if info:
                    pos = info['position']
                    size = info['size']
                    self.capture_region = (pos[0], pos[1], pos[0] + size[0], pos[1] + size[1])
                    return self.capture_screen_region()
                return None
        except Exception as e:
            logger.error(f"Error capturing window: {e}")
            raise CaptureError(f"Window capture failed: {e}")
    
    def capture_screen_region(self) -> Optional[np.ndarray]:
        """Capture screen region."""
        if not self.capture_region:
            return None
            
        try:
            return self.backend.capture_screen_region(self.capture_region)
        except Exception as e:
            logger.error(f"Error capturing screen region: {e}")
            raise CaptureError(f"Region capture failed: {e}")
    
    def get_window_info(self) -> Optional[Dict]:
        """Get window info (maintains backward compatibility)."""
        if not self.target_window_handle:
            return None
            
        try:
            info = self.backend.get_window_info_by_handle(self.target_window_handle)
            if info:
                return {
                    'title': info.title,
                    'handle': info.handle,
                    'size': info.size,
                    'position': info.position,
                    'visible': info.visible
                }
            return None
        except Exception as e:
            logger.error(f"Error getting window info: {e}")
            return None
    
    def is_window_valid(self) -> bool:
        """Check if target window is valid."""
        if not self.target_window_handle:
            return False
            
        try:
            return self.backend.is_window_valid(self.target_window_handle)
        except Exception as e:
            logger.error(f"Error checking window validity: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        self.target_window_handle = None
        self.capture_region = None
        logger.info("Capture cleanup completed")