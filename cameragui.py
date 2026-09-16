import cv2
import tkinter as tk
from PIL import Image, ImageTk

# --- Global Variables ---
cap = None
current_frame = None
capture = 0

def init_camera():
    """Initializes the Raspberry Pi camera or standard webcam."""
    global cap
    cap = cv2.VideoCapture(0) # 0 is usually the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")

def update_video_feed(video_label):
    """Grabs a frame from the camera, converts it, and updates the GUI."""
    global cap, current_frame
    
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            # Save the raw BGR frame globally so the capture button can access it
            current_frame = frame.copy()
            
            # Convert BGR (OpenCV format) to RGB (Tkinter/Pillow format)
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Using pillow as tinkter by itself does not support Images.
            pil_image = Image.fromarray(cv2_image)
            # Putting the Image on tkinkter Gui
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update the label with the new image
            video_label.tk_image = tk_image # Prevent garbage collection
            video_label.configure(image=tk_image)
            
    # Schedule this function to run again in 15 milliseconds (~60 fps limit)
    video_label.after(15, update_video_feed, video_label)

def capture_photo():
    """Triggered when the user clicks the capture button."""
    global current_frame, capture
    if current_frame is not None:
        filename = f"testingoutcomes/raw/capture{capture}.png"
        cv2.imwrite(filename, current_frame)
        print(f"Success! Photo saved as {filename}")
    else:
        print("Error: No frame available to capture.")

    capture =+ 1

def cleanup(root):
    """Releases the camera and closes the window properly."""
    global cap
    print("Shutting down camera...")
    if cap is not None:
        cap.release() # Shutting the camera off.
    root.destroy() # Shutting GUI

def main():
    # 1. Start the camera
    init_camera()
    
    # 2. Setup the Tkinter window
    root = tk.Tk()
    root.title("DocuNode - Camera Test")
    root.geometry("800x600")
    
    # 3. Create a label to hold the video feed
    video_label = tk.Label(root)
    video_label.pack(pady=10)
    
    # 4. Create the capture button
    capture_btn = tk.Button(root, text="Click Photo", command=capture_photo, 
                            font=("Arial", 16, "bold"), bg="blue", fg="white", padx=20, pady=10)
    capture_btn.pack(pady=10)
    
    # 5. Bind the window close button to our cleanup function
    root.protocol("WM_DELETE_WINDOW", lambda: cleanup(root))
    
    # 6. Kick off the video loop
    update_video_feed(video_label)
    
    # 7. Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()