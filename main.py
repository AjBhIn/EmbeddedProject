
import os
import cv2
import time

# Tell OpenCV to send the window to TigerVNC (usually :1 or :0)
os.environ["DISPLAY"] = ":0"  

def initialize_directories():
    """
    Creates local folder structures on the Pi for organizing raw captures and processed scans.
    """
    directories = ["testingoutcomes/raw", "testingoutcomes/processed"]
    for path in directories:
        os.makedirs(path, exist_ok=True)
        print(f"[+] Storage directory ready: {path}")

def check_sense_hat():
    """
    Validates connection to the Sense HAT and flashes the 8x8 LED matrix green to test LEDs.
    """
    try:
        from sense_hat import SenseHat
        import time
        
        sense = SenseHat()
        # Briefly flash green to verify matrix LED connectivity
        sense.clear(0, 255, 0)
        time.sleep(0.5)
        sense.clear()
        
        print("[+] Sense HAT interface verified.")
        return True
    except ImportError:
        print("[!] Sense HAT library not installed. (Testing on standard PC?)")
        return False
    except Exception as e:
        print(f"[-] Sense HAT hardware connection failed: {e}")
        return False

def check_camera_feed(output_file="scans/raw/test_capture.jpg"):
    """
    Connects to a USB webcam using V4L2, allows warm-up frames, and saves a test image.
    """
    # Try video index 0 first
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # If index 0 fails, fallback to index 1 (common for USB webcams on Pi)
    if not cap.isOpened():
        print("[!] Index 0 unavailable. Attempting camera index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("[-] Error: Logitech webcam not found on index 0 or 1.")
        return False

    # Set explicit resolution to prevent backend buffer mismatches
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Allow sensor to initialize auto-exposure and white balance
    time.sleep(1)
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()  # Free hardware resource

    if ret and frame is not None:
        cv2.imwrite(output_file, frame)
        print(f"[+] Logitech webcam verified! Saved test frame to: {output_file}")
        return True
    else:
        print("[-] Error: Camera stream opened, but failed to retrieve frame pixels.")
        return False

if __name__ == "__main__":
    print("--- Running Phase 1 Initialization Checks ---")
    initialize_directories()
    check_sense_hat()
    check_camera_feed()