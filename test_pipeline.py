import cv2
import easyocr
import numpy as np
import os
import re

def preprocess_and_read_plate(crop, reader):
    """
    Optimized OCR preprocessing and left-to-right reading:
    1. Upscale crop to optimal character height (180px - 300px).
    2. Convert to grayscale + apply CLAHE for balanced contrast.
    3. Bilateral filter for noise reduction without blurring edges.
    4. EasyOCR reading with alphanumeric allowlist.
    5. Left-to-right geometric sorting of detected text fragments.
    """
    if crop is None or crop.size == 0:
        return "", 0.0

    img_h, img_w = crop.shape[:2]
    # Upscale crop so height is at least 180px
    scale = max(2.0, 200.0 / img_h) if img_h > 0 else 2.5
    resized = cv2.resize(crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 7, 50, 50)

    # First pass: Enhanced Grayscale
    passes = [
        ("Enhanced Grayscale", denoised),
        ("Original Resized", resized),
        ("Otsu Threshold", cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    ]

    best_text = ""
    best_conf = 0.0

    for name, img_pass in passes:
        results = reader.readtext(
            img_pass,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
            paragraph=False,
            detail=1
        )
        if not results:
            continue

        # Sort all detected text fragments strictly from LEFT to RIGHT
        sorted_res = sorted(results, key=lambda r: min(pt[0] for pt in r[0]))
        text_parts = []
        conf_parts = []

        for bbox, txt, conf in sorted_res:
            clean = re.sub(r'[^A-Za-z0-9]', '', txt)
            if clean and conf > 0.15:
                text_parts.append(clean)
                conf_parts.append(conf)

        if text_parts:
            full_text = " ".join(text_parts).upper()
            avg_conf = float(np.mean(conf_parts))
            if avg_conf > best_conf and len(re.sub(r'[^A-Za-z0-9]', '', full_text)) >= 2:
                best_conf = avg_conf
                best_text = full_text

        # Early exit if we already have a strong reading
        if best_conf > 0.60 and len(re.sub(r'[^A-Za-z0-9]', '', best_text)) >= 4:
            break

    return best_text, best_conf

def detect_and_crop_plate_robust(image, reader, plate_cascade=None):
    img_h, img_w = image.shape[:2]
    candidates = []

    # 1. EasyOCR direct text localization
    ocr_full = reader.readtext(image)
    if ocr_full:
        # Sort full image detections left-to-right
        sorted_ocr = sorted(ocr_full, key=lambda r: min(pt[0] for pt in r[0]))
        # Combine nearby bounding boxes if they form a multi-word plate
        plate_boxes = []
        for bbox, text, conf in sorted_ocr:
            clean_t = re.sub(r'[^A-Za-z0-9]', '', text)
            if len(clean_t) >= 2 and conf > 0.15:
                pts = np.array(bbox, dtype=np.int32)
                bx, by, bw, bh = cv2.boundingRect(pts)
                plate_boxes.append((bx, by, bw, bh))

        if plate_boxes:
            # Union of detected plate text bounding boxes
            min_x = max(0, min(b[0] for b in plate_boxes))
            min_y = max(0, min(b[1] for b in plate_boxes))
            max_x = min(img_w, max(b[0] + b[2] for b in plate_boxes))
            max_y = min(img_h, max(b[1] + b[3] for b in plate_boxes))

            pw = max_x - min_x
            ph = max_y - min_y

            # Pad bounding box to include the full license plate frame
            pad_x = int(pw * 0.25)
            pad_y = int(ph * 0.40)
            px = max(0, min_x - pad_x)
            py = max(0, min_y - pad_y)
            pw_padded = min(img_w - px, pw + 2 * pad_x)
            ph_padded = min(img_h - py, ph + 2 * pad_y)

            candidates.append(((px, py, pw_padded, ph_padded), "OCR Scene Localization"))

    # 2. Haar Cascade candidates
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if plate_cascade is not None:
        haar_boxes = plate_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=2, minSize=(40, 15))
        for (hx, hy, hw, hh) in haar_boxes:
            candidates.append(((int(hx), int(hy), int(hw), int(hh)), "Haar Cascade Classifier"))

    # 3. Contour analysis candidates
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

    # Evaluate candidates
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
