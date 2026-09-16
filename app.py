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
# 3. ROBUST DETECTION & CROPPING PIPELINE
# ==========================================
def detect_and_crop_plate(image):
    """
    Robust vehicle number plate detection and cropping pipeline:
    1. Collects candidate bounding boxes using:
       a. OCR Text Localization (direct text region detection on vehicle)
       b. Haar Cascade Classifier (OpenCV pre-trained model)
       c. Morphological edge & contour analysis (rectangular aspect ratio)
    2. Validates & scores each candidate crop using EasyOCR character verification.
    3. Selects the candidate with the highest-confidence alphanumeric license plate text.
    4. Crops the exact plate region with padding and returns the annotated vehicle and crop.
    """
    img_h, img_w = image.shape[:2]
    candidates = []

    # Candidate 1: EasyOCR direct scene text localization
    ocr_full = reader.readtext(image)
    for bbox, text, conf in ocr_full:
        clean_t = re.sub(r'[^A-Za-z0-9]', '', text)
        if len(clean_t) >= 2 and conf > 0.15:
            pts = np.array(bbox, dtype=np.int32)
            bx, by, bw, bh = cv2.boundingRect(pts)
            # Add padding around text to capture the full plate frame
            pad_x = int(bw * 0.25)
            pad_y = int(bh * 0.35)
            px = max(0, bx - pad_x)
            py = max(0, by - pad_y)
            pw = min(img_w - px, bw + 2 * pad_x)
            ph = min(img_h - py, bh + 2 * pad_y)
            candidates.append(((px, py, pw, ph), "OCR Text Localization", text, conf))

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
            candidates.append(((int(hx), int(hy), int(hw), int(hh)), "Haar Cascade Classifier", "", 0.0))

    # Candidate 3: Morphological / Contour Edge Analysis
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / float(ch)
        if cy > img_h * 0.25 and 1.8 <= aspect_ratio <= 6.0 and 1200 <= (cw * ch) <= (img_w * img_h * 0.25):
            candidates.append(((int(cx), int(cy), int(cw), int(ch)), "Contour Analysis", "", 0.0))

    best_crop = None
    best_box = None
    best_text = ""
    best_score = -1.0
    best_method = ""

    # Evaluate & score each candidate crop
    seen_boxes = set()
    for (x, y, w, h), method, initial_text, initial_conf in candidates:
        box_key = (x // 10, y // 10, w // 10, h // 10)
        if box_key in seen_boxes:
            continue
        seen_boxes.add(box_key)

        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        if initial_text and initial_conf > 0.35:
            raw_text = initial_text
            conf = initial_conf
        else:
            crop_results = reader.readtext(crop)
            if crop_results:
                raw_text = " ".join([r[1] for r in crop_results]).strip()
                conf = float(np.mean([r[2] for r in crop_results]))
            else:
                raw_text = ""
                conf = 0.0

        clean_text = re.sub(r'[^A-Za-z0-9]', '', raw_text)
        text_len = len(clean_text)

        has_letters = bool(re.search(r'[A-Za-z]', clean_text))
        has_digits = bool(re.search(r'[0-9]', clean_text))

        score = conf * 2.5
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
# 4. SIDEBAR: SAMPLE IMAGES OR UPLOAD
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
# 5. MAIN PROCESSING PIPELINE & DISPLAY
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
