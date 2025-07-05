"""
Display renderer for hologram pyramid.
Handles fullscreen display, user input, and the main rendering loop with improved performance.
"""
import threading
import time
import logging
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class DisplayMode(Enum):
    """Display mode enumeration."""
    WINDOWED = "windowed"
    FULLSCREEN = "fullscreen"

class RenderState(Enum):
    """Rendering state enumeration."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"

@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    fps: float = 0.0
    frame_time_ms: float = 0.0
    dropped_frames: int = 0
    total_frames: int = 0

@dataclass
class RendererConfig:
    """Configuration for the renderer."""
    target_fps: int = 60
    show_fps: bool = True
    show_debug_info: bool = False
    vsync: bool = True
    frame_drop_threshold_ms: float = 33.0  # Drop frames taking longer than this
    max_consecutive_drops: int = 5

class PyramidRenderer:
    """Enhanced display renderer for pyramid hologram with improved performance and features."""

    def __init__(self, config: Optional[RendererConfig] = None):
        self.config = config or RendererConfig()
        self.window_name = "Hologram Pyramid Display"
        
        # State management
        self._state = RenderState.STOPPED
        self._display_mode = DisplayMode.WINDOWED
        
        # Threading and synchronization
        self.frame_lock = threading.Lock()
        self.current_frame: Optional[np.ndarray] = None
        self._shutdown_event = threading.Event()
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._frame_times = []
        self._consecutive_drops = 0
        
        # Display properties
        self._window_initialized = False
        self._display_width = 1920
        self._display_height = 1080
        
        logger.info(f"PyramidRenderer initialized with target FPS: {self.config.target_fps}")

    def get_monitor_resolution(self, monitor_index: int = 0) -> Tuple[int, int]:
        """Get the resolution of the specified monitor with fallback options."""
        try:
            # Try screeninfo first
            import screeninfo
            monitors = screeninfo.get_monitors()
            if monitor_index < len(monitors):
                monitor = monitors[monitor_index]
                resolution = (monitor.width, monitor.height)
                logger.info(f"Monitor {monitor_index} resolution: {resolution}")
                return resolution
            else:
                logger.warning(f"Monitor index {monitor_index} not found, using primary monitor")
                if monitors:
                    return (monitors[0].width, monitors[0].height)
        except ImportError:
            logger.warning("screeninfo not available, trying alternative methods")
        except Exception as e:
            logger.error(f"Error getting monitor resolution with screeninfo: {e}")
        
        # Try tkinter as fallback
        try:
            import tkinter as tk
            root = tk.Tk()
            resolution = (root.winfo_screenwidth(), root.winfo_screenheight())
            root.destroy()
            logger.info(f"Resolution from tkinter: {resolution}")
            return resolution
        except Exception as e:
            logger.warning(f"Error getting resolution with tkinter: {e}")
        
        # Ultimate fallback
        logger.warning("Using fallback resolution: 1920x1080")
        return (1920, 1080)

    def initialize_display(self, width: int, height: int) -> bool:
        """Initialize the display window with enhanced error handling."""
        try:
            if self._window_initialized:
                logger.warning("Display already initialized")
                return True
                
            self._display_width = width
            self._display_height = height
            
            # Create window with specific flags for better performance
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.window_name, width, height)
            
            # Set window position (center of screen)
            screen_w, screen_h = self.get_monitor_resolution()
            pos_x = max(0, (screen_w - width) // 2)
            pos_y = max(0, (screen_h - height) // 2)
            cv2.moveWindow(self.window_name, pos_x, pos_y)
            
            # Show initial black frame
            initial_frame = np.zeros((height, width, 3), dtype=np.uint8)
            self._add_startup_message(initial_frame)
            cv2.imshow(self.window_name, initial_frame)
            cv2.waitKey(1)
            
            self._window_initialized = True
            logger.info(f"Display initialized: {width}x{height} at position ({pos_x}, {pos_y})")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing display: {e}")
            return False

    def _add_startup_message(self, frame: np.ndarray):
        """Add startup message to the initial frame."""
        try:
            h, w = frame.shape[:2]
            
            # Add centered text
            messages = [
                "Hologram Pyramid Display",
                "Initializing...",
                "",
                "Controls:",
                "F - Fullscreen | Q/ESC - Quit | Space - Pause"
            ]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (255, 255, 255)
            thickness = 2
            
            # Calculate text positioning
            line_height = 40
            start_y = h // 2 - (len(messages) * line_height) // 2
            
            for i, message in enumerate(messages):
                if message:  # Skip empty lines
                    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
                    x = (w - text_size[0]) // 2
                    y = start_y + i * line_height
                    cv2.putText(frame, message, (x, y), font, font_scale, color, thickness)
                    
        except Exception as e:
            logger.error(f"Error adding startup message: {e}")

    def start_rendering(self, frame_callback: Callable[[], Optional[np.ndarray]]):
        """Start the main rendering loop with enhanced performance monitoring."""
        if not self._window_initialized:
            logger.error("Display not initialized. Call initialize_display() first.")
            return
            
        self._state = RenderState.RUNNING
        self._shutdown_event.clear()
        frame_time = 1.0 / self.config.target_fps
        
        logger.info("Starting rendering loop")
        
        try:
            while self._state != RenderState.STOPPED and not self._shutdown_event.is_set():
                loop_start_time = time.time()
                
                # Check if window was closed
                if not self._is_window_open():
                    logger.info("Window closed by user")
                    break
                
                # Update frame if not paused
                if self._state == RenderState.RUNNING:
                    self._update_frame(frame_callback)
                
                # Render current frame
                self._render_frame()
                
                # Handle input
                self._handle_input()
                
                # Performance management
                self._manage_frame_timing(loop_start_time, frame_time)
                
        except Exception as e:
            logger.error(f"Error in rendering loop: {e}")
            raise
        finally:
            self._state = RenderState.STOPPED
            logger.info("Rendering loop stopped")

    def _is_window_open(self) -> bool:
        """Check if the OpenCV window is still open."""
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except:
            return False

    def _update_frame(self, frame_callback: Callable[[], Optional[np.ndarray]]):
        """Update the current frame using the callback."""
        try:
            frame_start = time.time()
            new_frame = frame_callback()
            frame_time = (time.time() - frame_start) * 1000
            
            if new_frame is not None:
                with self.frame_lock:
                    self.current_frame = new_frame.copy()
                    
                # Track frame timing
                self._track_frame_performance(frame_time)
            else:
                # Callback returned None (e.g., source unavailable)
                if self._state == RenderState.RUNNING:
                    logger.warning("Frame callback returned None")
                    self.stop_rendering()
                    
        except Exception as e:
            logger.error(f"Error updating frame: {e}")

    def _render_frame(self):
        """Render the current frame to the display."""
        try:
            if self.current_frame is not None:
                with self.frame_lock:
                    display_frame = self.current_frame.copy()
                
                # Add overlay information
                self._add_overlay_info(display_frame)
                
                # Display the frame
                cv2.imshow(self.window_name, display_frame)
                
                self.metrics.total_frames += 1
                
        except Exception as e:
            logger.error(f"Error rendering frame: {e}")

    def _add_overlay_info(self, frame: np.ndarray):
        """Add FPS and debug information to the frame."""
        try:
            if not (self.config.show_fps or self.config.show_debug_info):
                return
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (0, 255, 0)
            thickness = 2
            
            y_offset = 30
            line_height = 30
            
            if self.config.show_fps:
                fps_text = f"FPS: {self.metrics.fps:.1f}"
                cv2.putText(frame, fps_text, (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
            
            if self._state == RenderState.PAUSED:
                cv2.putText(frame, "PAUSED", (10, y_offset), font, font_scale, (0, 0, 255), thickness)
                y_offset += line_height
            
            if self.config.show_debug_info:
                debug_info = [
                    f"Frame Time: {self.metrics.frame_time_ms:.1f}ms",
                    f"Dropped: {self.metrics.dropped_frames}",
                    f"Mode: {self._display_mode.value}",
                    f"Resolution: {self._display_width}x{self._display_height}"
                ]
                
                for info in debug_info:
                    cv2.putText(frame, info, (10, y_offset), font, 0.5, (255, 255, 0), 1)
                    y_offset += 20
                    
        except Exception as e:
            logger.error(f"Error adding overlay info: {e}")

    def _handle_input(self):
        """Process keyboard inputs with enhanced key handling."""
        try:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                logger.info("Quit key pressed")
                self.stop_rendering()
            elif key == ord('f'):
                self._toggle_fullscreen()
            elif key == ord(' '):
                self._toggle_pause()
            elif key == ord('r'):
                self._restart_capture()
            elif key == ord('d'):
                self._toggle_debug_info()
            elif key == ord('s'):
                self._save_screenshot()
                
        except Exception as e:
            logger.error(f"Error handling input: {e}")

    def _toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        try:
            if self._display_mode == DisplayMode.WINDOWED:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                self._display_mode = DisplayMode.FULLSCREEN
                logger.info("Switched to fullscreen mode")
            else:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                self._display_mode = DisplayMode.WINDOWED
                logger.info("Switched to windowed mode")
                
        except Exception as e:
            logger.error(f"Error toggling fullscreen: {e}")

    def _toggle_pause(self):
        """Toggle the pause state."""
        try:
            if self._state == RenderState.RUNNING:
                self._state = RenderState.PAUSED
                logger.info("Rendering paused")
            elif self._state == RenderState.PAUSED:
                self._state = RenderState.RUNNING
                logger.info("Rendering resumed")
                
        except Exception as e:
            logger.error(f"Error toggling pause: {e}")

    def _restart_capture(self):
        """Signal to restart capture (placeholder for future implementation)."""
        logger.info("Restart capture requested")
        # This could be extended to signal the main app to reinitialize capture

    def _toggle_debug_info(self):
        """Toggle debug information display."""
        self.config.show_debug_info = not self.config.show_debug_info
        logger.info(f"Debug info {'enabled' if self.config.show_debug_info else 'disabled'}")

    def _save_screenshot(self):
        """Save current frame as screenshot."""
        try:
            if self.current_frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pyramid_screenshot_{timestamp}.png"
                cv2.imwrite(filename, self.current_frame)
                logger.info(f"Screenshot saved as {filename}")
                
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")

    def _track_frame_performance(self, frame_time_ms: float):
        """Track frame performance metrics."""
        try:
            self.metrics.frame_time_ms = frame_time_ms
            
            # Check for dropped frames
            if frame_time_ms > self.config.frame_drop_threshold_ms:
                self.metrics.dropped_frames += 1
                self._consecutive_drops += 1
                
                if self._consecutive_drops > self.config.max_consecutive_drops:
                    logger.warning(f"High frame drop rate detected: {self._consecutive_drops} consecutive drops")
            else:
                self._consecutive_drops = 0
                
            # Store recent frame times for analysis
            self._frame_times.append(frame_time_ms)
            if len(self._frame_times) > 100:  # Keep last 100 frames
                self._frame_times.pop(0)
                
        except Exception as e:
            logger.error(f"Error tracking performance: {e}")

    def _manage_frame_timing(self, loop_start_time: float, target_frame_time: float):
        """Manage frame timing and FPS calculation."""
        try:
            # Calculate FPS
            self._fps_counter += 1
            current_time = time.time()
            
            if current_time - self._fps_timer >= 1.0:
                self.metrics.fps = self._fps_counter
                self._fps_counter = 0
                self._fps_timer = current_time
            
            # Frame rate limiting
            elapsed = current_time - loop_start_time
            wait_time = target_frame_time - elapsed
            
            if wait_time > 0:
                # Convert to milliseconds for cv2.waitKey
                wait_ms = max(1, int(wait_time * 1000))
                cv2.waitKey(wait_ms)
            elif wait_time < -target_frame_time:
                # Frame took too long, consider dropping
                logger.debug(f"Frame timing exceeded target by {-wait_time:.3f}s")
                
        except Exception as e:
            logger.error(f"Error managing frame timing: {e}")

    def stop_rendering(self):
        """Signal the rendering loop to stop."""
        self._state = RenderState.STOPPED
        self._shutdown_event.set()
        logger.info("Stop rendering requested")

    def is_running(self) -> bool:
        """Check if renderer is currently running."""
        return self._state != RenderState.STOPPED

    def is_paused(self) -> bool:
        """Check if renderer is paused."""
        return self._state == RenderState.PAUSED

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics

    def get_average_frame_time(self) -> float:
        """Get average frame time from recent frames."""
        if not self._frame_times:
            return 0.0
        return sum(self._frame_times) / len(self._frame_times)

    def reset_performance_metrics(self):
        """Reset performance counters."""
        self.metrics = PerformanceMetrics()
        self._frame_times.clear()
        self._consecutive_drops = 0
        logger.info("Performance metrics reset")

    def set_target_fps(self, fps: int):
        """Update target FPS."""
        if 1 <= fps <= 120:
            self.config.target_fps = fps
            logger.info(f"Target FPS updated to {fps}")
        else:
            logger.warning(f"Invalid FPS value: {fps}. Must be between 1 and 120")

    def cleanup(self):
        """Clean up resources and close windows."""
        try:
            logger.info("Cleaning up renderer resources")
            
            # Stop rendering if still running
            if self.is_running():
                self.stop_rendering()
                
            # Wait a moment for cleanup
            time.sleep(0.1)
            
            # Destroy OpenCV windows
            cv2.destroyAllWindows()
            
            # Clear frame data
            with self.frame_lock:
                self.current_frame = None
                
            self._window_initialized = False
            logger.info("Renderer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during renderer cleanup: {e}")

    def create_info_overlay(self, frame: np.ndarray, info_text: str, position: str = "top_left"):
        """Create an information overlay on the frame."""
        try:
            h, w = frame.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)
            thickness = 2
            
            # Get text dimensions
            text_size = cv2.getTextSize(info_text, font, font_scale, thickness)[0]
            
            # Calculate position
            if position == "top_left":
                x, y = 10, 30
            elif position == "top_right":
                x, y = w - text_size[0] - 10, 30
            elif position == "bottom_left":
                x, y = 10, h - 10
            elif position == "bottom_right":
                x, y = w - text_size[0] - 10, h - 10
            elif position == "center":
                x, y = (w - text_size[0]) // 2, h // 2
            else:
                x, y = 10, 30  # Default to top_left
                
            # Add background rectangle for better readability
            padding = 5
            cv2.rectangle(frame, 
                         (x - padding, y - text_size[1] - padding),
                         (x + text_size[0] + padding, y + padding),
                         (0, 0, 0), -1)
            
            # Add text
            cv2.putText(frame, info_text, (x, y), font, font_scale, color, thickness)
            
        except Exception as e:
            logger.error(f"Error creating info overlay: {e}")

    def set_window_title(self, title: str):
        """Set custom window title."""
        try:
            self.window_name = title
            if self._window_initialized:
                cv2.setWindowTitle(self.window_name, title)
            logger.info(f"Window title set to: {title}")
        except Exception as e:
            logger.error(f"Error setting window title: {e}")

    def export_performance_report(self, filename: str = "performance_report.txt"):
        """Export performance metrics to a file."""
        try:
            with open(filename, 'w') as f:
                f.write("Hologram Pyramid Display - Performance Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Current FPS: {self.metrics.fps:.2f}\n")
                f.write(f"Target FPS: {self.config.target_fps}\n")
                f.write(f"Average Frame Time: {self.get_average_frame_time():.2f}ms\n")
                f.write(f"Total Frames: {self.metrics.total_frames}\n")
                f.write(f"Dropped Frames: {self.metrics.dropped_frames}\n")
                f.write(f"Drop Rate: {(self.metrics.dropped_frames / max(1, self.metrics.total_frames) * 100):.2f}%\n")
                f.write(f"Display Resolution: {self._display_width}x{self._display_height}\n")
                f.write(f"Display Mode: {self._display_mode.value}\n")
                
                if self._frame_times:
                    f.write(f"\nFrame Time Statistics:\n")
                    f.write(f"Min: {min(self._frame_times):.2f}ms\n")
                    f.write(f"Max: {max(self._frame_times):.2f}ms\n")
                    f.write(f"Avg: {self.get_average_frame_time():.2f}ms\n")
                    
            logger.info(f"Performance report exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction