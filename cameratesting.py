import os
import cv2
import time

# Route GUI output to TigerVNC desktop session
os.environ["DISPLAY"] = ":0"

def capture_test_photo():
    os.makedirs("testingoutcomes/raw", exist_ok=True)
    output_path = "testingoutcomes/raw/1_raw_capture.jpg"
    
    # Initialize Logitech C270 on video index 0
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("[-] Could not access camera on /dev/video0")
        return False

    # Force MJPEG mode and standard 720p resolution supported by the C270
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Allow auto-exposure and white balance to warm up
    time.sleep(1)
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        cv2.imwrite(output_path, frame)
        print(f"[+] Capture successful! Saved to: {output_path}")
        
        # Display window on TigerVNC preview
        cv2.imshow("Raw Capture Preview", frame)
        print("[i] Press any key on the VNC window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
    else:
        print("[-] Failed to capture valid frame pixels.")
        return False

if __name__ == "__main__":
    capture_test_photo()