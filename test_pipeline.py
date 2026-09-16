import cv2
import easyocr
import numpy as np
import os
import re

def detect_and_crop_plate_robust(image, reader, plate_cascade=None):
    img_h, img_w = image.shape[:2]
    candidates = []

    # 1. EasyOCR direct text localization (Finds exact text boxes on vehicle)
    # EasyOCR's CRAFT detector locates text regions in natural scene images
    ocr_full = reader.readtext(image)
    for bbox, text, conf in ocr_full:
        clean_t = re.sub(r'[^A-Za-z0-9]', '', text)
        if len(clean_t) >= 2 and conf > 0.15:
            # bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            pts = np.array(bbox, dtype=np.int32)
            bx, by, bw, bh = cv2.boundingRect(pts)
            # Add padding around the text to capture the full plate border
            pad_x = int(bw * 0.25)
            pad_y = int(bh * 0.35)
            px = max(0, bx - pad_x)
            py = max(0, by - pad_y)
            pw = min(img_w - px, bw + 2 * pad_x)
            ph = min(img_h - py, bh + 2 * pad_y)
            candidates.append(((px, py, pw, ph), "OCR Text Localization", text, conf))

    # 2. Haar Cascade candidates
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if plate_cascade is not None:
        haar_boxes = plate_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=2, minSize=(40, 15))
        for (hx, hy, hw, hh) in haar_boxes:
            candidates.append(((hx, hy, hw, hh), "Haar Cascade Classifier", "", 0.0))

    # 3. Contour analysis candidates
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / float(ch)
        if cy > img_h * 0.25 and 1.8 <= aspect_ratio <= 6.0 and 1200 <= (cw * ch) <= (img_w * img_h * 0.25):
            candidates.append(((cx, cy, cw, ch), "Contour Analysis", "", 0.0))

    best_crop = None
    best_box = None
    best_text = ""
    best_score = -1.0
    best_method = ""

    # Evaluate each candidate crop
    seen_boxes = set()
    for (x, y, w, h), method, initial_text, initial_conf in candidates:
        # Avoid duplicate evaluations of nearly identical boxes
        box_key = (x // 10, y // 10, w // 10, h // 10)
        if box_key in seen_boxes:
            continue
        seen_boxes.add(box_key)

        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        # If we already have OCR text from full image localization, use it
        if initial_text and initial_conf > 0.3:
            raw_text = initial_text
            conf = initial_conf
        else:
            # Run OCR on the crop
            crop_results = reader.readtext(crop)
            if crop_results:
                raw_text = " ".join([r[1] for r in crop_results]).strip()
                conf = np.mean([r[2] for r in crop_results])
            else:
                raw_text = ""
                conf = 0.0

        clean_text = re.sub(r'[^A-Za-z0-9]', '', raw_text)
        text_len = len(clean_text)

        # Plate scoring logic:
        # - Alphanumeric characters count (plates have 3-10 chars)
        # - Has both letters and digits -> bonus
        # - OCR confidence
        has_letters = bool(re.search(r'[A-Za-z]', clean_text))
        has_digits = bool(re.search(r'[0-9]', clean_text))

        score = conf * 2.0
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

        # Preference for realistic plate aspect ratio (2.0 to 5.0)
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

def test_robust_pipeline():
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    
    for img_name in sorted(os.listdir('test_images')):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')): continue
        img_path = os.path.join('test_images', img_name)
        img = cv2.imread(img_path)
        
        annotated, crop, box, method, text = detect_and_crop_plate_robust(img, reader, cascade)
        print(f"[{img_name}] Method: {method} | Box: {box} | Plate Text: '{text}'")

if __name__ == "__main__":
    test_robust_pipeline()
