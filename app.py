import os
import re
import cv2
import easyocr
import numpy as np
import streamlit as st

# ==========================================
# 1. PAGE CONFIG & HEADER
# ==========================================
st.set_page_config(
    page_title="Vehicle Number Plate Recognition",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Number Plate Recognition System")
st.markdown(
    "**Workflow:** Upload Vehicle Image ➔ Detect Plate ➔ Crop Plate Region ➔ EasyOCR Reading ➔ Display Plate Number"
)
st.write("---")

# ==========================================
# 2. LOAD EASYOCR & HAAR CASCADE (CACHED)
# ==========================================
@st.cache_resource
def load_ocr_reader():
    """Load the EasyOCR English reader into memory once."""
    return easyocr.Reader(['en'], gpu=False, verbose=False)

@st.cache_resource
def load_haar_cascade():
    """Load the OpenCV Haar Cascade classifier for license plates."""
    cascade_path = "haarcascade_russian_plate_number.xml"
    if os.path.exists(cascade_path):
        return cv2.CascadeClassifier(cascade_path)
    return None

reader = load_ocr_reader()
plate_cascade = load_haar_cascade()

# ==========================================
# 3. OCR PREPROCESSING & DISAMBIGUATION
# ==========================================
def clean_plate_text(raw_text):
    """
    Cleans OCR output and resolves common character ambiguities:
    - Strips unwanted non-alphanumeric noise.
    - Fixes broken loop misclassifications (e.g. 'J' misread for '0'/'O' in 'VJ16' -> 'V016').
    """
    cleaned = re.sub(r'[^A-Za-z0-9 ]', '', raw_text).strip().upper()
    tokens = cleaned.split()

    fixed_tokens = []
    for token in tokens:
        t = token
        # Replace 'J' when placed between an initial letter and numbers (e.g. 'VJ16' -> 'V016')
        t = re.sub(r'(^[A-Z])J(\d)', r'\g<1>0\2', t)
        # Replace leading 'J' before numbers if misread
        t = re.sub(r'^J(\d{2,})', r'0\1', t)
        fixed_tokens.append(t)

    return " ".join(fixed_tokens)

def preprocess_and_read_plate(crop, ocr_reader):
    """
    Enhanced OCR preprocessing & Left-to-Right Geometric Reading:
    1. Cubic upscaling to optimal character stroke resolution.
    2. Unsharp masking + CLAHE for balanced contrast and closed character loops.
    3. Morphological closing to repair broken '0'/'O' curves.
    4. Left-to-right geometric sorting of detected text blocks.
    5. Disambiguation filter for license plate alphanumeric formatting.
    """
    if crop is None or crop.size == 0:
        return "", 0.0

    img_h, img_w = crop.shape[:2]
    # Upscale crop so height is at least 180px
    scale = max(2.0, 200.0 / img_h) if img_h > 0 else 2.5
    resized = cv2.resize(crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement & unsharp masking
    blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
    unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(unsharp)
    denoised = cv2.bilateralFilter(enhanced, 7, 50, 50)

    # Morphological closing to seal open loops in digits/letters
    adaptive_thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 9
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    passes = [
        ("Enhanced Grayscale", denoised),
        ("Adaptive Morph Closed", closed_thresh),
        ("Original Resized", resized)
    ]

    best_text = ""
    best_conf = 0.0

    for name, img_pass in passes:
        results = ocr_reader.readtext(
            img_pass,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
            paragraph=False,
            detail=1,
            contrast_ths=0.1,
            adjust_contrast=0.7,
            mag_ratio=1.5
        )
        if not results:
            continue

        # Sort all detected text fragments strictly from LEFT to RIGHT by min-x coordinate
        sorted_res = sorted(results, key=lambda r: min(pt[0] for pt in r[0]))
        text_parts = []
        conf_parts = []

        for bbox, txt, conf in sorted_res:
            clean = re.sub(r'[^A-Za-z0-9]', '', txt)
            if clean and conf > 0.15:
                text_parts.append(clean)
                conf_parts.append(conf)

        if text_parts:
            raw_text = " ".join(text_parts).upper()
            processed_text = clean_plate_text(raw_text)
            avg_conf = float(np.mean(conf_parts))

            if avg_conf > best_conf and len(re.sub(r'[^A-Za-z0-9]', '', processed_text)) >= 2:
                best_conf = avg_conf
                best_text = processed_text

        # Early exit if strong reading is achieved
        if best_conf > 0.65 and len(re.sub(r'[^A-Za-z0-9]', '', best_text)) >= 4:
            break

    return best_text, best_conf

# ==========================================
# 4. ROBUST DETECTION & CROPPING PIPELINE
# ==========================================
def detect_and_crop_plate(image):
    """
    Robust vehicle number plate detection and cropping pipeline:
    1. Collects candidate bounding boxes from:
       a. OCR Text Region Localization
       b. Haar Cascade Classifier
       c. Morphological Edge & Contour Analysis
    2. Validates & scores each candidate crop using the enhanced OCR preprocessor.
    3. Selects the candidate with the highest-confidence alphanumeric license plate text.
    4. Crops the exact plate region with proportional padding.
    """
    img_h, img_w = image.shape[:2]
    candidates = []

    # Candidate 1: EasyOCR direct scene text localization
    ocr_full = reader.readtext(image)
    if ocr_full:
        sorted_ocr = sorted(ocr_full, key=lambda r: min(pt[0] for pt in r[0]))
        plate_boxes = []
        for bbox, text, conf in sorted_ocr:
            clean_t = re.sub(r'[^A-Za-z0-9]', '', text)
            if len(clean_t) >= 2 and conf > 0.15:
                pts = np.array(bbox, dtype=np.int32)
                bx, by, bw, bh = cv2.boundingRect(pts)
                plate_boxes.append((bx, by, bw, bh))

        if plate_boxes:
            min_x = max(0, min(b[0] for b in plate_boxes))
            min_y = max(0, min(b[1] for b in plate_boxes))
            max_x = min(img_w, max(b[0] + b[2] for b in plate_boxes))
            max_y = min(img_h, max(b[1] + b[3] for b in plate_boxes))

            pw = max_x - min_x
            ph = max_y - min_y

            # Pad bounding box to capture the full plate frame
            pad_x = int(pw * 0.25)
            pad_y = int(ph * 0.40)
            px = max(0, min_x - pad_x)
            py = max(0, min_y - pad_y)
            pw_padded = min(img_w - px, pw + 2 * pad_x)
            ph_padded = min(img_h - py, ph + 2 * pad_y)

            candidates.append(((px, py, pw_padded, ph_padded), "OCR Scene Localization"))

    # Candidate 2: Haar Cascade Classifier
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if plate_cascade is not None:
        haar_boxes = plate_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.08, 
            minNeighbors=2, 
            minSize=(40, 15)
        )
        for (hx, hy, hw, hh) in haar_boxes:
            candidates.append(((int(hx), int(hy), int(hw), int(hh)), "Haar Cascade Classifier"))

    # Candidate 3: Morphological / Contour Edge Analysis
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / float(ch)
        if cy > img_h * 0.25 and 1.8 <= aspect_ratio <= 6.0 and 1500 <= (cw * ch) <= (img_w * img_h * 0.25):
            candidates.append(((int(cx), int(cy), int(cw), int(ch)), "Contour Analysis"))

    best_crop = None
    best_box = None
    best_text = ""
    best_score = -1.0
    best_method = ""

    # Evaluate candidates with enhanced OCR preprocessor
    seen_boxes = set()
    for (x, y, w, h), method in candidates[:6]:
        box_key = (x // 15, y // 15, w // 15, h // 15)
        if box_key in seen_boxes:
            continue
        seen_boxes.add(box_key)

        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        raw_text, conf = preprocess_and_read_plate(crop, reader)
        clean_text = re.sub(r'[^A-Za-z0-9]', '', raw_text)
        text_len = len(clean_text)

        has_letters = bool(re.search(r'[A-Za-z]', clean_text))
        has_digits = bool(re.search(r'[0-9]', clean_text))

        score = conf * 3.0
        if 3 <= text_len <= 10:
            score += 3.0
            if has_letters and has_digits:
                score += 2.0
            elif has_letters or has_digits:
                score += 1.0
        elif text_len > 10:
            score += 1.0
        else:
            score -= 2.0

        ar = w / float(h)
        if 2.0 <= ar <= 5.5:
            score += 1.0

        if score > best_score and text_len >= 2:
            best_score = score
            best_crop = crop
            best_box = (x, y, w, h)
            best_text = raw_text.upper()
            best_method = method

    if best_box is not None:
        x, y, w, h = best_box
        annotated = image.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
        return annotated, best_crop, best_box, best_method, best_text

    return image, None, None, None, ""

# ==========================================
# 5. SIDEBAR: SAMPLE IMAGES OR UPLOAD
# ==========================================
st.sidebar.header("📁 Image Selection")
option = st.sidebar.radio("Choose Input Method:", ["Upload Custom Image", "Use Sample Vehicle Image"])

image_to_process = None

if option == "Upload Custom Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Vehicle Image", 
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

else:
    sample_files = sorted([f for f in os.listdir("test_images") if f.endswith(('.jpg', '.png', '.jpeg'))])
    selected_sample = st.sidebar.selectbox("Select a sample test vehicle:", sample_files)
    if selected_sample:
        sample_path = os.path.join("test_images", selected_sample)
        image_to_process = cv2.imread(sample_path)

# ==========================================
# 6. MAIN PROCESSING PIPELINE & DISPLAY
# ==========================================
if image_to_process is not None:
    image_rgb = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Vehicle Input Image")
        st.image(image_rgb, caption="Input Vehicle", use_container_width=True)

    with col2:
        st.subheader("2. Number Plate Recognition Results")
        with st.spinner("Detecting number plate and running OCR..."):
            annotated_img, cropped_plate, box, method, plate_text = detect_and_crop_plate(image_to_process)

        if cropped_plate is not None:
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            cropped_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)

            st.write(f"✅ **Detection Method:** `{method}`")
            st.write(f"📍 **Plate Location (x, y, w, h):** `{box}`")

            # Display cropped plate
            st.image(cropped_rgb, caption="Extracted / Cropped Plate Region", width=260)

            st.write("---")
            st.subheader("3. Detected Plate Number:")
            if plate_text:
                st.success(f"### 🔤 **`{plate_text}`**")
            else:
                st.warning("⚠️ Plate region cropped, but characters could not be recognized clearly.")

        else:
            st.error("❌ No number plate could be detected in this image. Please ensure the vehicle plate is clearly visible.")

else:
    st.info("👈 Please select a sample image or upload a vehicle photo from the sidebar to begin recognition.")
