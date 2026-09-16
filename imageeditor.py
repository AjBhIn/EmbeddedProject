import cv2
import numpy as np
from rembg import remove, new_session
import os

def remove_background(input_path, output_path):
    print(f"[1] Loading AI to remove background from {input_path}...")
    session = new_session("u2netp")
    
    with open(input_path, 'rb') as i:
        input_data = i.read()
        
    print("[2] Processing image...")
    output_data = remove(input_data, session=session)
    
    with open(output_path, 'wb') as o:
        o.write(output_data)
    print(f"[3] Background removed! Saved temporary file to: {output_path}")

def order_points(pts):
    """
    Bulletproof mathematical corner sorting.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum of X and Y: Smallest is Top-Left, Largest is Bottom-Right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Difference of Y minus X: Smallest is Top-Right, Largest is Bottom-Left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def straighten_and_pad(image_path, final_output_path):
    print("[4] Pinning corners and flattening the card...")
    
    # Load image. IMREAD_UNCHANGED ensures we capture the alpha (transparency) channel if it exists.
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 1. BULLETPROOF MASK GENERATION
    if img.shape[2] == 4:
        # It's a PNG with transparency. Use the alpha channel as a pure mask.
        _, mask = cv2.threshold(img[:, :, 3], 200, 255, cv2.THRESH_BINARY)
        # Convert image to standard BGR for later processing
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        # It's a JPG without transparency. Assume black background.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        img_bgr = img
    
    # 2. Find the card's exact boundaries
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[-] Error: No object found in image.")
        return False
        
    c = max(contours, key=cv2.contourArea)
    
    # 3. Calculate mathematically perfect corners of the card
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # Order the corners strictly (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    ordered_pts = order_points(box.astype("float32"))
    (tl, tr, br, bl) = ordered_pts
    
    # 4. Measure the exact width and height in pixels
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Create the destination template (a perfectly flat rectangle)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    # 5. Warp the raw image to completely fill the flat rectangle, destroying all background
    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    flat_img = cv2.warpPerspective(img_bgr, M, (maxWidth, maxHeight))
    
    # 6. Smart Rotation: Force Landscape
    if flat_img.shape[0] > flat_img.shape[1]:
        print("[5] Portrait orientation detected: Rotating to Landscape.")
        flat_img = cv2.rotate(flat_img, cv2.ROTATE_90_CLOCKWISE)
        
    # 7. Add 100px pure white padding
    print("[6] Adding 100px padding...")
    padding = 100
    final_padded = cv2.copyMakeBorder(
        flat_img, padding, padding, padding, padding, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
    )
    
    cv2.imwrite(final_output_path, final_padded)
    print(f"[+] Success! Final padded image saved to: {final_output_path}")
    return True


def process_document(input_file, final_output_file):
    """
    Bridge function: The camera GUI calls this function.
    It automatically handles the temporary files and runs your pipeline.
    """
    print(f"\n--- Starting Vision Pipeline for {input_file} ---")
    temp_file = "temp_nobg.png"
    
    try:
        # Step 1: Run your background removal
        remove_background(input_file, temp_file)
        
        # Step 2: Run your math, rotation, and padding
        success = straighten_and_pad(temp_file, final_output_file)
        
        # Step 3: Clean up the temporary file so it doesn't clutter your folder
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print("[7] Temporary files cleaned up.")
            
        return success
        
    except Exception as e:
        print(f"[-] Fatal error in vision pipeline: {e}")
        return False

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    RAW_IMAGE = "test_capture_2.png"    
    TEMP_IMAGE = "testingoutcomes/raw/temp_nobg.png"         
    FINAL_IMAGE = "testingoutcomes/processed/final_padded.jpg"     
    
    if os.path.exists(RAW_IMAGE):
        remove_background(RAW_IMAGE, TEMP_IMAGE)
        straighten_and_pad(TEMP_IMAGE, FINAL_IMAGE)
    else:
        print(f"[-] Error: Could not find '{RAW_IMAGE}'.")