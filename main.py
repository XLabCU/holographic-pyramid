"""
Main application for Hologram Pyramid Display.
Combines capture, processing, and rendering for real-time holographic display.
"""
import argparse
import sys
import numpy as np
import cv2
import time

from core.capture import WindowCapture
from core.processor import ImageProcessor
from core.renderer import PyramidRenderer


class HologramPyramidApp:
    """Main application class for hologram pyramid display."""

    def __init__(self, args):
        self.args = args
        self.capture = WindowCapture()
        # Initialize processor with a placeholder resolution; will be updated later.
        self.processor = ImageProcessor(1920, 1080)
        self.renderer = PyramidRenderer()
        self.selected_window_info = None

    def list_windows(self):
        """List available windows for capture."""
        print("\nAvailable windows:")
        print("-" * 50)
        windows = self.capture.get_available_windows()
        for i, window in enumerate(windows):
            title = window.get('title', 'N/A')
            size = window.get('size', ('N/A', 'N/A'))
            print(f"{i+1:2d}. {title} ({size[0]}x{size[1]})")
        return windows

    def select_window_interactive(self) -> bool:
        """Interactive window selection."""
        windows = self.list_windows()
        if not windows:
            print("No suitable windows found!")
            return False

        while True:
            try:
                choice = input(f"Select window (1-{len(windows)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return False

                index = int(choice) - 1
                if 0 <= index < len(windows):
                    selected = windows[index]
                    
                    # Attempt to set the window handle in the capture class
                    if self.capture.set_target_window_by_handle(selected['handle']):
                        # If successful, use the 'selected' dictionary we already have.
                        # This is more robust than re-querying.
                        self.selected_window_info = selected 
                        print(f"Selected: {self.selected_window_info['title']}")
                        return True
                    else:
                        print("Failed to set target window. It might have closed. Please try another.")
                else:
                    print("Invalid selection. Please try again.")
            except (ValueError, KeyboardInterrupt):
                print("\nInvalid input or interrupted. Exiting selection.")
                return False

    def run(self):
        """Main application execution flow."""
        print("Hologram Pyramid Display")
        print("=" * 40)

        frame_callback = None

        # --- Setup capture source based on arguments ---
        if self.args.demo:
            print("Running in Demo Mode...")
            frame_callback = self._get_demo_frame
        elif self.args.region:
            print(f"Capturing screen region: {self.args.region}")
            x, y, w, h = self.args.region
            self.capture.set_capture_region(x, y, w, h)
            frame_callback = self._get_region_frame
        elif self.args.title:
            print(f"Attempting to capture window with title: '{self.args.title}'")
            if not self.capture.set_target_window(self.args.title):
                print(f"Error: Could not find a window with title containing '{self.args.title}'.")
                return
            self.selected_window_info = self.capture.get_window_info()
            frame_callback = self._get_window_frame
        else: # Interactive mode
            if not self.args.demo and not self.args.region and not self.selected_window_info:
             if not self.select_window_interactive():
                 print("No window selected. Exiting.")
                 return
             frame_callback = self._get_window_frame

        # --- Initialize display and processor ---
        print("\nInitializing display...")
        # Use primary monitor resolution for the display window
        screen_width, screen_height = self.renderer.get_monitor_resolution()
        self.processor.set_output_resolution(screen_width, screen_height)
        
        if not self.renderer.initialize_display(screen_width, screen_height):
            print("Failed to initialize display. Exiting.")
            return

        print("[DEBUG] Pausing for 0.5 seconds before starting renderer...")
        time.sleep(0.5)

        print("\nStarting hologram display...")
        print("Controls:")
        print("  F - Toggle fullscreen")
        print("  Q or ESC - Quit")
        print("  Space - Pause/Resume")
        
        try:
            self.renderer.start_rendering(frame_callback)
        except KeyboardInterrupt:
            print("\nShutting down by user request...")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
        finally:
            self.cleanup()

    def _get_window_frame(self):
        """Callback to capture a window frame."""
        if not self.capture.is_window_valid():
            print("Target window has been closed or is no longer valid. Shutting down.")
            self.renderer.stop_rendering()
            return None
        raw_frame = self.capture.capture_window()
        if raw_frame is None:
            return None
        return self.processor.create_pyramid_layout(raw_frame)

    def _get_region_frame(self):
        """Callback to capture a screen region."""
        raw_frame = self.capture.capture_screen_region()
        if raw_frame is None:
            return None
        return self.processor.create_pyramid_layout(raw_frame)

    def _get_demo_frame(self):
        """Callback to generate a demo frame."""
        # Create test pattern
        test_image = np.zeros((400, 600, 3), dtype=np.uint8)
        # Add some colored rectangles for testing
        test_image[50:150, 50:150] = [255, 0, 0]    # Red square
        test_image[50:150, 200:300] = [0, 255, 0]   # Green square
        test_image[200:300, 50:150] = [0, 0, 255]   # Blue square
        test_image[200:300, 200:300] = [255, 255, 0] # Yellow square
        cv2.putText(test_image, "DEMO MODE", (150, 350),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return self.processor.create_pyramid_layout(test_image)

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources.")
        self.renderer.cleanup()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time Hologram Pyramid Display")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--title', type=str, help="Title of the window to capture.")
    group.add_argument('--region', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                       help="Capture a specific screen region (x, y, width, height).")
    group.add_argument('--demo', action='store_true', help="Run in demo mode with a test pattern.")
    
    # Add a note for macOS users
    parser.epilog = "For macOS, window capture may require screen recording permissions. If title capture fails, --region mode is a reliable alternative."

    args = parser.parse_args()
    
    app = HologramPyramidApp(args)
    app.run()

if __name__ == "__main__":
    main()