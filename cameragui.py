import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
import math
import threading

# --- Hardware Setup ---
try:
    from sense_hat import SenseHat
    sense = SenseHat()
    SENSE_AVAILABLE = True
except ImportError:
    print("Warning: Sense HAT not detected. Mocking stability data.")
    SENSE_AVAILABLE = False
    sense = None

# --- Global Variables ---
cap = None
current_frame = None
photo_count = 0
app_state = "LIVE"  # States: "LIVE", "REVIEW", "PROCESSING"

# WDT & Auto-Capture Config
is_scanning = False
scan_start_time = 0
good_frame_start_time = 0
WDT_TIMEOUT = 10.0          # Seconds before WDT stops the scan
AUTO_CAPTURE_DELAY = 1.5    # Seconds conditions must be perfect
last_accel = {'x': 0, 'y': 0, 'z': 0}


# --- Camera & Sensor Functions ---
def init_camera():
    global cap
    cap = cv2.VideoCapture(0) # Change to 1 or -1 if 0 fails on Pi
    if not cap.isOpened():
        print("Error: Could not open camera.")

def check_focus(frame, threshold=50.0): # Relaxed threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return focus_score, focus_score > threshold

def check_framing(frame, min_area_ratio=0.10): # Relaxed threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, False
        
    largest_contour = max(contours, key=cv2.contourArea)
    doc_area = cv2.contourArea(largest_contour)
    
    height, width = frame.shape[:2]
    area_ratio = doc_area / (height * width)
    return area_ratio, area_ratio > min_area_ratio

def check_stability(threshold=0.2): # Relaxed threshold
    global last_accel
    if not SENSE_AVAILABLE:
        return 0.0, True 

    acceleration = sense.get_accelerometer_raw()
    x, y, z = acceleration['x'], acceleration['y'], acceleration['z']
    dx = abs(x - last_accel['x'])
    dy = abs(y - last_accel['y'])
    dz = abs(z - last_accel['z'])
    
    movement = dx + dy + dz
    last_accel = {'x': x, 'y': y, 'z': z}
    return movement, movement < threshold


# --- App State Control Functions ---
def start_scan(instruction_lbl):
    global is_scanning, scan_start_time, good_frame_start_time
    is_scanning = True
    scan_start_time = time.time()
    good_frame_start_time = 0 
    instruction_lbl.configure(text="Auto-Scan started... Please hold still.", fg="blue")

def capture_photo(instruction_lbl, capture_btn, force_btn, process_btn, retry_btn):
    """Saves raw photo, stops live feed, and shows review buttons."""
    global current_frame, photo_count, is_scanning, app_state
    
    photo_count += 1 
    if current_frame is not None:
        raw_filename = f"raw_capture_{photo_count}.png"
        cv2.imwrite(raw_filename, current_frame)
        
        # Change state to REVIEW
        app_state = "REVIEW"
        is_scanning = False
        
        instruction_lbl.configure(text=f"Photo taken! Does this look good?", fg="blue")
        
        # Swap buttons
        capture_btn.pack_forget()  
        force_btn.pack_forget()
        
        process_btn.configure(command=lambda: confirm_processing(raw_filename, instruction_lbl, capture_btn, force_btn, process_btn, retry_btn))
        process_btn.pack(pady=5)   
        retry_btn.pack(pady=5)     

def retry_scan(instruction_lbl, capture_btn, force_btn, process_btn, retry_btn):
    """Returns to LIVE state and brings back capture buttons."""
    global app_state, is_scanning
    app_state = "LIVE"
    is_scanning = False
    
    instruction_lbl.configure(text="Scan cancelled. Ready for new photo.", fg="black")
    
    process_btn.pack_forget() 
    retry_btn.pack_forget()   
    capture_btn.pack(pady=10) 
    force_btn.pack(pady=10)   

def confirm_processing(raw_filename, instruction_lbl, capture_btn, force_btn, process_btn, retry_btn):
    """Hides buttons and starts the multithreaded image processing."""
    global app_state
    app_state = "PROCESSING"
    
    process_btn.pack_forget()
    retry_btn.pack_forget()
    
    editor_thread = threading.Thread(
        target=process_in_background, 
        args=(raw_filename, instruction_lbl, capture_btn, force_btn)
    )
    editor_thread.start()

def process_in_background(input_filename, instruction_lbl, capture_btn, force_btn):
    """Runs imageeditor.py without freezing the GUI."""
    global app_state
    try:
        instruction_lbl.configure(text=f"Processing {input_filename}...\nThis may take a few seconds.", fg="orange")
        
        # --- VISION PIPELINE TRIGGER ---
        import imageeditor # Make sure this file is in the same folder!
        output_filename = f"processed_{input_filename}"
        
        # Call the main function of your imageeditor file. 
        # (If your function isn't called process_document, change it here)
        imageeditor.process_document(input_filename, output_filename)
        
        instruction_lbl.configure(text=f"Done!\nSaved as {output_filename}", fg="green")
    except Exception as e:
        print(f"Error: {e}")
        instruction_lbl.configure(text="Error processing image.", fg="red")
    finally:
        app_state = "LIVE"
        capture_btn.pack(pady=10)
        force_btn.pack(pady=10)


# --- Video Loop ---
def update_video_feed(video_label, focus_lbl, frame_lbl, stable_lbl, instruction_lbl, capture_btn, force_btn, process_btn, retry_btn):
    global cap, current_frame, is_scanning, scan_start_time, good_frame_start_time, app_state
    
    if app_state == "LIVE" and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            current_frame = frame.copy()
            
            # 1. Metrics
            focus_score, is_focused = check_focus(frame) 
            frame_ratio, is_framed = check_framing(frame)
            move_score, is_stable = check_stability()
            
            focus_lbl.configure(text=f"Focus: {focus_score:.0f}", fg="green" if is_focused else "red")
            frame_lbl.configure(text=f"Size: {frame_ratio*100:.0f}%", fg="green" if is_framed else "red")
            stable_lbl.configure(text=f"Move: {move_score:.3f}", fg="green" if is_stable else "red")
            
            # 2. Auto-Capture WDT Logic
            if is_scanning:
                current_time = time.time()
                
                if (current_time - scan_start_time) > WDT_TIMEOUT:
                    is_scanning = False
                    instruction_lbl.configure(text="WDT TIMEOUT:\nCould not get a good frame.", fg="red")
                
                elif is_focused and is_framed and is_stable:
                    if good_frame_start_time == 0:
                        good_frame_start_time = current_time
                        instruction_lbl.configure(text="Locking on... Hold still!", fg="orange")
                    
                    elif (current_time - good_frame_start_time) >= AUTO_CAPTURE_DELAY:
                        capture_photo(instruction_lbl, capture_btn, force_btn, process_btn, retry_btn)
                else:
                    good_frame_start_time = 0
                    instruction_lbl.configure(text="Adjust document to make all metrics green.", fg="blue")

            # 3. Draw to GUI
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            video_label.tk_image = tk_image
            video_label.configure(image=tk_image)
            
    video_label.after(15, update_video_feed, video_label, focus_lbl, frame_lbl, stable_lbl, instruction_lbl, capture_btn, force_btn, process_btn, retry_btn)

def cleanup(root):
    global cap
    if cap is not None: cap.release()
    root.destroy()


# --- Main GUI Setup ---
def main():
    init_camera()
    root = tk.Tk()
    root.title("DocuNode - Camera System")
    root.geometry("900x600")
    
    main_container = tk.Frame(root)
    main_container.pack(fill="both", expand=True, padx=10, pady=10)
    
    video_label = tk.Label(main_container)
    video_label.pack(side="left", padx=10)
    
    control_panel = tk.Frame(main_container)
    control_panel.pack(side="right", fill="y", padx=20, pady=20)
    
    # UI Labels
    focus_label = tk.Label(control_panel, text="Focus: --", font=("Arial", 14, "bold"))
    focus_label.pack(pady=10)
    frame_label = tk.Label(control_panel, text="Size: --", font=("Arial", 14, "bold"))
    frame_label.pack(pady=10)
    stable_label = tk.Label(control_panel, text="Movement: --", font=("Arial", 14, "bold"))
    stable_label.pack(pady=10)
    instruction_label = tk.Label(control_panel, text="Ready. Click 'Start' or 'Force Capture'.", font=("Arial", 12, "italic"), wraplength=250)
    instruction_label.pack(pady=20)
    
    # Buttons
    capture_btn = tk.Button(control_panel, text="Start Auto-Scan", font=("Arial", 14, "bold"), bg="blue", fg="white", padx=20, pady=5)
    force_btn = tk.Button(control_panel, text="Force Capture", font=("Arial", 14, "bold"), bg="purple", fg="white", padx=20, pady=5)
    process_btn = tk.Button(control_panel, text="Process Image", font=("Arial", 16, "bold"), bg="green", fg="white", padx=20, pady=10)
    retry_btn = tk.Button(control_panel, text="Retry", font=("Arial", 14, "bold"), bg="gray", fg="white", padx=20, pady=5)
    
    capture_btn.configure(command=lambda: start_scan(instruction_label))
    force_btn.configure(command=lambda: capture_photo(instruction_label, capture_btn, force_btn, process_btn, retry_btn))
    retry_btn.configure(command=lambda: retry_scan(instruction_label, capture_btn, force_btn, process_btn, retry_btn))
    
    # Only pack the initial live buttons
    capture_btn.pack(pady=10)
    force_btn.pack(pady=10)
    
    root.protocol("WM_DELETE_WINDOW", lambda: cleanup(root))
    update_video_feed(video_label, focus_label, frame_label, stable_label, instruction_label, capture_btn, force_btn, process_btn, retry_btn)
    root.mainloop()

if __name__ == "__main__":
    main()