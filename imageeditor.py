"""
DocuNode Vision Pipeline Module (imageeditor.py)
------------------------------------------------
Handles edge-computed perspective scanning steps:
1. AI Background Removal (rembg / u2netp model)
2. Contour & Corner Detection (OpenCV)
3. Corner Sorting (Sum and Difference Method)
4. Perspective Transformation (Warp Matrix)
5. Landscape Rotation & 100px Alpha Padding
"""

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

# Global session cache to avoid re-loading the ONNX model into memory on every scan
_REMBG_SESSION = None

def get_rembg_session():
    """Lazily initializes and caches the u2netp lightweight model session."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        # u2netp is optimized for low-power edge devices like Raspberry Pi
        _REMBG_SESSION = new_session(model_name="u2netp")
    return _REMBG_SESSION

def preload_engine():
    """
    Warms up the numba JIT compiler and rembg model session in memory.
    Call this on application startup in a background thread.
    """
    session = get_rembg_session()
    # Create a tiny 10x10 dummy image to force numba JIT compilation
    dummy = Image.new("RGB", (10, 10), color="white")
    remove(dummy, session=session)
    return True

def sort_corners(pts):
    """
    Sorts 4 corner points into strict order:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    Prevents diagonal scrambling using the Sum and Difference method.
    """
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")

    # Sum Method: Top-Left has smallest sum, Bottom-Right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    # Difference Method: Top-Right has smallest diff (y - x), Bottom-Left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    return rect

def process_document(input_path, output_path):
    """
    Executes the full sequentially ordered vision processing pipeline.
    
    Parameters:
        input_path (str): File path of raw captured image.
        output_path (str): File path to save final processed .png document.
    """
    # -------------------------------------------------------------------------
    # STEP 1: AI Background Removal
    # -------------------------------------------------------------------------
    session = get_rembg_session()
    input_img = Image.open(input_path)
    
    # Strip background to leave document on transparent alpha channel
    output_png = remove(input_img, session=session)
    img_np = np.array(output_png)

    # Extract Alpha Channel to locate document outline
    if img_np.shape[2] == 4:
        alpha = img_np[:, :, 3]
    else:
        alpha = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # -------------------------------------------------------------------------
    # STEP 2: Contour & Corner Detection
    # -------------------------------------------------------------------------
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No document contour detected after background removal.")

    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    # Fallback to minimum area bounding box if 4 distinct corners are not found
    if len(approx) == 4:
        doc_pts = approx.reshape(4, 2)
    else:
        rect = cv2.minAreaRect(largest_contour)
        doc_pts = cv2.boxPoints(rect)

    # -------------------------------------------------------------------------
    # STEP 3: Bulletproof Corner Sorting
    # -------------------------------------------------------------------------
    rect_corners = sort_corners(doc_pts)
    (tl, tr, br, bl) = rect_corners

    # -------------------------------------------------------------------------
    # STEP 4: Perspective Transformation
    # -------------------------------------------------------------------------
    # Calculate output rectangle width
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Calculate output rectangle height
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Flat destination coordinates
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Map skewed physical corners to perfectly flat rectangle
    M = cv2.getPerspectiveTransform(rect_corners, dst)
    warped = cv2.warpPerspective(img_np, M, (max_width, max_height))

    # -------------------------------------------------------------------------
    # STEP 5: Smart Rotation & Transparent Padding
    # -------------------------------------------------------------------------
    h, w = warped.shape[:2]
    # Force output image into landscape orientation
    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    # Add 100px completely transparent border ([0, 0, 0, 0] RGBA)
    padded = cv2.copyMakeBorder(
        warped, 
        top=100, bottom=100, left=100, right=100, 
        borderType=cv2.BORDER_CONSTANT, 
        value=[0, 0, 0, 0]
    )

    # Save final PNG output
    final_pil = Image.fromarray(padded)
    final_pil.save(output_path, "PNG")
    return True