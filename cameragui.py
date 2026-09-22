"""
DocuNode Control Panel GUI (cameragui.py)
----------------------------------------
CustomTkinter interface featuring a 2-row layout:
- Row 0: Live camera feed display using lightweight ImageTk
- Row 1: Simplified live status badges (Focus, Size, Stability), 
         status advice banner, and control buttons.

Includes background pre-loading of imageeditor.py to ensure zero lag.
"""

import time
import threading
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

# --- Sense HAT Hardware Import (Graceful Fallback) ---
try:
    from sense_hat import SenseHat
    sense = SenseHat()
    SENSE_AVAILABLE = True
except ImportError:
    print("Warning: Sense HAT hardware not found. Accelerometer checks will be simulated.")
    SENSE_AVAILABLE = False
    sense = None

# =============================================================================
# THRESHOLD CALIBRATION & CONFIGURATION SETTINGS
# Adjust these constants to tune system sensitivity on the Raspberry Pi
# =============================================================================

# 1. Focus Threshold (Variance of Laplacian)
# Higher value = requires sharper focus. Lower = more forgiving with soft lenses.
FOCUS_THRESHOLD = 100.0  

# 2. Framing/Size Area Ratio
# Minimum percentage of the total camera frame that the document contour must occupy (0.15 = 15%).
MIN_DOC_AREA_RATIO = 0.15  

# 3. Motion/Stability Threshold (Accelerometer Delta)
# Polled from Sense HAT. Lower value = requires desk to be completely still. Higher = allows minor vibrations.
STABILITY_THRESHOLD = 0.05  

# 4. Watchdog Timer (WDT) Timeout (in seconds)
# Maximum allowed scan time before triggering setup advice to the user on the UI/LED matrix[cite: 1].
WDT_TIMEOUT = 10.0  

# 5. Continuous Stability Lock Duration (in seconds)
# How long all 3 conditions (Focus, Size, Motion) must continuously pass before auto-triggering capture.
AUTO_CAPTURE_DELAY = 1.5  

# =============================================================================
# GLOBAL STATE VARIABLES
# =============================================================================
app_state = "LIVE"  # States: "LIVE", "REVIEW", "PROCESSING"
engine_ready = False
cap = None
current_frame = None
captured_frame = None
photo_count = 0

scan_start_time = 0
good_frame_start_time = 0
last_accel = {'x': 0, 'y': 0, 'z': 0}

# Configure CustomTkinter appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# =============================================================================
# LIVE FEED VALIDATION FUNCTIONS ("THE BOUNCER")
# =============================================================================

def check_focus(frame, threshold=FOCUS_THRESHOLD):
    """Calculates image sharpness using OpenCV Variance of the Laplacian."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Comparison against mathematical FOCUS_THRESHOLD
    return focus_score >= threshold

def check_framing(frame, min_area_ratio=MIN_DOC_AREA_RATIO):
    """Checks if a document contour occupies sufficient frame area."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
        
    largest_contour = max(contours, key=cv2.contourArea)
    doc_area = cv2.contourArea(largest_contour)
    frame_area = frame.shape[0] * frame.shape[1]
    
    area_ratio = doc_area / frame_area
    # Comparison against mathematical MIN_DOC_AREA_RATIO
    return area_ratio >= min_area_ratio

def check_stability(threshold=STABILITY_THRESHOLD):
    """Polls Sense HAT accelerometer to verify table isn't shaking[cite: 1]."""
    global last_accel
    if not SENSE_AVAILABLE:
        return True

    accel = sense.get_accelerometer_raw()
    dx = abs(accel['x'] - last_accel['x'])
    dy = abs(accel['y'] - last_accel['y'])
    dz = abs(accel['z'] - last_accel['z'])
    movement = dx + dy + dz
    
    last_accel = accel
    # Comparison against mathematical STABILITY_THRESHOLD
    return movement < threshold

# =============================================================================
# ASYNCHRONOUS ENGINE PRE-LOADER & WORKERS
# =============================================================================

def start_engine_preload_thread(advice_lbl, root_window):
    """Launches a silent background thread to warm up numba/rembg on startup."""
    def worker():
        global engine_ready
        try:
            import imageeditor
            imageeditor.preload_engine()
            engine_ready = True
            root_window.after(0, lambda: advice_lbl.configure(
                text="System Ready. Align document under camera.", text_color="#D1D5DB"
            ))
        except Exception as e:
            print(f"Preload warning: {e}")
            engine_ready = True  # Proceed anyway

    threading.Thread(target=worker, daemon=True).start()

def action_process_image(advice_lbl, root_window):
    """Launches imageeditor processing thread without locking the GUI[cite: 1]."""
    global app_state, photo_count
    
    app_state = "PROCESSING"
    advice_lbl.configure(text="Processing image with rembg pipeline... Please wait.", text_color="#2196F3")
    
    input_file = f"raw_capture_{photo_count}.png"
    output_file = f"processed_doc_{photo_count}.png"
    
    threading.Thread(
        target=run_vision_pipeline_thread, 
        args=(input_file, output_file, advice_lbl, root_window), 
        daemon=True
    ).start()

def run_vision_pipeline_thread(input_path, output_path, advice_lbl, root_window):
    """Executes imageeditor.py tasks and updates UI safely via root.after()."""
    try:
        import imageeditor
        imageeditor.process_document(input_path, output_path)
        
        # Safe thread transition back to main GUI thread
        root_window.after(0, lambda: advice_lbl.configure(
            text=f"Success! Processed scan saved to {output_path}", text_color="#4CAF50"
        ))
    except Exception as e:
        root_window.after(0, lambda: advice_lbl.configure(
            text=f"Pipeline Error: {str(e)}", text_color="#F44336"
        ))

# =============================================================================
# APPLICATION CONTROLS & STATE MANAGEMENT
# =============================================================================

def init_camera():
    """Initializes OpenCV video stream."""
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera device could not be opened.")

def update_video_feed(video_label, focus_lbl, frame_lbl, stable_lbl, advice_lbl, btn_process, btn_retry):
    """Main feed loop handling live validation, state transitions, and rendering."""
    global cap, current_frame, captured_frame, app_state
    global scan_start_time, good_frame_start_time
    
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            # 1. LIVE FEED SCANNING STATE
            if app_state == "LIVE":
                current_frame = frame.copy()
                
                # Perform live metric checks (Returns purely Booleans)
                is_focused = check_focus(frame)
                is_framed = check_framing(frame)
                is_stable = check_stability()
                
                # Update live metric status badges with clean text labels
                focus_lbl.configure(
                    text="Focus: OK" if is_focused else "Focus: BLUR",
                    fg_color="#1E3A1E" if is_focused else "#3A1E1E",
                    text_color="#4CAF50" if is_focused else "#F44336"
                )

                frame_lbl.configure(
                    text="Size: OK" if is_framed else "Size: SMALL",
                    fg_color="#1E3A1E" if is_framed else "#3A1E1E",
                    text_color="#4CAF50" if is_framed else "#F44336"
                )

                stable_lbl.configure(
                    text="Stable: OK" if is_stable else "Stable: SHAKY",
                    fg_color="#1E3A1E" if is_stable else "#3A1E1E",
                    text_color="#4CAF50" if is_stable else "#F44336"
                )
                
                # Watchdog Timer & Auto-Capture Decision Logic
                current_time = time.time()
                if (current_time - scan_start_time) > WDT_TIMEOUT:
                    advice_lbl.configure(
                        text="WDT Advice: Adjust desk lighting or move document closer.",
                        text_color="#FF9800"
                    )
                elif is_focused and is_framed and is_stable:
                    if good_frame_start_time == 0:
                        good_frame_start_time = current_time
                        advice_lbl.configure(text="Locking on... Hold steady!", text_color="#2196F3")
                    elif (current_time - good_frame_start_time) >= AUTO_CAPTURE_DELAY:
                        trigger_review_mode(advice_lbl, btn_process, btn_retry)
                else:
                    good_frame_start_time = 0
                    if engine_ready:
                        if not is_framed:
                            advice_lbl.configure(text="Advice: Center document in camera view.", text_color="#D1D5DB")
                        elif not is_focused:
                            advice_lbl.configure(text="Advice: Camera blurry. Increase lighting.", text_color="#D1D5DB")
                        elif not is_stable:
                            advice_lbl.configure(text="Advice: Desk movement detected. Hold still.", text_color="#D1D5DB")

                display_frame = current_frame

            # 2. REVIEW & PROCESSING STATE (Displays static captured photo)
            else:
                display_frame = captured_frame if captured_frame is not None else frame

            # Render frame using standard ImageTk (High performance on Raspberry Pi)
            cv2_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_img)
            pil_img = pil_img.resize((640, 360), Image.Resampling.NEAREST)
            
            tk_img = ImageTk.PhotoImage(image=pil_img)
            video_label.tk_image = tk_img
            video_label.configure(image=tk_img)

    # Schedule next loop pass (~30 FPS execution) using local parameter names
    video_label.after(30, update_video_feed, video_label, focus_lbl, frame_lbl, stable_lbl, advice_lbl, btn_process, btn_retry)
    
def trigger_review_mode(advice_lbl, btn_process, btn_retry):
    """Freezes live feed and saves initial raw capture."""
    global app_state, current_frame, captured_frame, photo_count
    
    app_state = "REVIEW"
    captured_frame = current_frame.copy()
    photo_count += 1
    
    filename = f"raw_capture_{photo_count}.png"
    cv2.imwrite(filename, captured_frame)
    
    advice_lbl.configure(text=f"Frame captured as {filename}. Ready to process?", text_color="#4CAF50")
    btn_process.configure(state="normal", fg_color="#2EA043")
    btn_retry.configure(state="normal", fg_color="#D97706")

def action_force_capture(advice_lbl, btn_process, btn_retry):
    """Manual button override for developer testing."""
    trigger_review_mode(advice_lbl, btn_process, btn_retry)

def action_retry_scan(advice_lbl, btn_process, btn_retry):
    """Resets interface back to live feed mode."""
    global app_state, scan_start_time, good_frame_start_time
    
    app_state = "LIVE"
    scan_start_time = time.time()
    good_frame_start_time = 0
    
    advice_lbl.configure(text="Scanning live stream...", text_color="#D1D5DB")
    btn_process.configure(state="disabled", fg_color="#374151")
    btn_retry.configure(state="disabled", fg_color="#374151")

def cleanup(root):
    """Cleanly releases hardware camera resources on application shutdown."""
    global cap
    if cap is not None:
        cap.release()
    root.destroy()

# =============================================================================
# MAIN WINDOW GUI LAYOUT (2-ROW GRID)
# =============================================================================

def main():
    global scan_start_time
    init_camera()
    scan_start_time = time.time()
    
    root = ctk.CTk()
    root.title("DocuNode Edge Appliance - Control Panel")
    root.geometry("820x680")
    root.resizable(False, False)

    # 2-Row Grid Setup
    root.grid_rowconfigure(0, weight=3)  # Row 0: Camera Feed Panel
    root.grid_rowconfigure(1, weight=2)  # Row 1: Controls & Metrics Panel
    root.grid_columnconfigure(0, weight=1)

    # -------------------------------------------------------------------------
    # ROW 0: VIDEO DISPLAY FRAME
    # -------------------------------------------------------------------------
    video_frame = ctk.CTkFrame(root, corner_radius=12, fg_color="#1F2937")
    video_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
    video_frame.grid_columnconfigure(0, weight=1)
    video_frame.grid_rowconfigure(0, weight=1)

    video_label = ctk.CTkLabel(video_frame, text="", corner_radius=8)
    video_label.grid(row=0, column=0, padx=10, pady=10)

    # -------------------------------------------------------------------------
    # ROW 1: CONTROLS & METRICS PANEL
    # -------------------------------------------------------------------------
    controls_panel = ctk.CTkFrame(root, corner_radius=12, fg_color="#111827")
    controls_panel.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")
    
    controls_panel.grid_rowconfigure(0, weight=1)  # Metric Badges Sub-row
    controls_panel.grid_rowconfigure(1, weight=1)  # Status Advice Banner Sub-row
    controls_panel.grid_rowconfigure(2, weight=1)  # Action Buttons Sub-row
    controls_panel.grid_columnconfigure(0, weight=1)

    # --- Metric Badges Sub-Row ---
    metrics_frame = ctk.CTkFrame(controls_panel, fg_color="transparent")
    metrics_frame.grid(row=0, column=0, padx=15, pady=5, sticky="ew")
    metrics_frame.grid_columnconfigure((0, 1, 2), weight=1)

    focus_badge = ctk.CTkLabel(
        metrics_frame, text="Focus: --", font=("Arial", 14, "bold"),
        corner_radius=8, height=35, fg_color="#374151"
    )
    focus_badge.grid(row=0, column=0, padx=5, sticky="ew")

    frame_badge = ctk.CTkLabel(
        metrics_frame, text="Size: --", font=("Arial", 14, "bold"),
        corner_radius=8, height=35, fg_color="#374151"
    )
    frame_badge.grid(row=0, column=1, padx=5, sticky="ew")

    stable_badge = ctk.CTkLabel(
        metrics_frame, text="Stable: --", font=("Arial", 14, "bold"),
        corner_radius=8, height=35, fg_color="#374151"
    )
    stable_badge.grid(row=0, column=2, padx=5, sticky="ew")

    # --- Status Advice Sub-Row ---
    advice_label = ctk.CTkLabel(
        controls_panel, text="Warming up processing engine...", 
        font=("Arial", 14, "italic"), text_color="#D1D5DB"
    )
    advice_label.grid(row=1, column=0, padx=15, pady=5)

    # --- Action Buttons Sub-Row ---
    buttons_frame = ctk.CTkFrame(controls_panel, fg_color="transparent")
    buttons_frame.grid(row=2, column=0, padx=15, pady=(5, 15), sticky="ew")
    buttons_frame.grid_columnconfigure((0, 1, 2), weight=1)

    btn_force = ctk.CTkButton(
        buttons_frame, text="Force Capture", font=("Arial", 14, "bold"),
        height=40, fg_color="#2563EB", hover_color="#1D4ED8",
        command=lambda: action_force_capture(advice_label, btn_process, btn_retry)
    )
    btn_force.grid(row=0, column=0, padx=5, sticky="ew")

    btn_retry = ctk.CTkButton(
        buttons_frame, text="Retry Scan", font=("Arial", 14, "bold"),
        height=40, fg_color="#374151", hover_color="#4B5563", state="disabled",
        command=lambda: action_retry_scan(advice_label, btn_process, btn_retry)
    )
    btn_retry.grid(row=0, column=1, padx=5, sticky="ew")

    btn_process = ctk.CTkButton(
        buttons_frame, text="Process Image", font=("Arial", 14, "bold"),
        height=40, fg_color="#374151", hover_color="#166534", state="disabled",
        command=lambda: action_process_image(advice_label, root)
    )
    btn_process.grid(row=0, column=2, padx=5, sticky="ew")

    # Bind application close handler
    root.protocol("WM_DELETE_WINDOW", lambda: cleanup(root))

    # Kick off background pre-loader thread
    start_engine_preload_thread(advice_label, root)

    # Kick off live video feed loop
    update_video_feed(video_label, focus_badge, frame_badge, stable_badge, advice_label, btn_process, btn_retry)

    root.mainloop()

if __name__ == "__main__":
    main()