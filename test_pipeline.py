import cv2
import easyocr
import numpy as np
import os
import re

def enhance_plate_for_ocr(crop):
    """
    Applies multi-scale enhancement and morphological repair to cropped plate images:
    - Upscaling (INTER_CUBIC)
    - Unsharp masking to accentuate fine character strokes
    - CLAHE for contrast balancing
    - Morphological closing to seal open loops (e.g. preventing '0'/'O' from breaking into 'J')
    - Adaptive Gaussian thresholding
    """
    img_h, img_w = crop.shape[:2]
    scale = max(2.5, 220.0 / img_h) if img_h > 0 else 3.0
    resized = cv2.resize(crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 1. Unsharp Masking (Crisp character boundary definition)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
    unsharp = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)

    # 2. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(unsharp)
    denoised = cv2.bilateralFilter(enhanced, 7, 50, 50)

    # 3. Morphological Closing on Threshold (reconnects broken loops in '0', 'O', '8', 'B')
    adaptive_thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 9
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    otsu_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return [
        ("Enhanced Grayscale", denoised),
        ("Unsharp Grayscale", unsharp),
        ("Adaptive Morph Closed", closed_thresh),
        ("Otsu Threshold", otsu_thresh),
        ("Original Resized", resized)
    ]

def clean_plate_text(raw_text):
    """
    Cleans OCR output and fixes common character misrecognitions on license plates:
    - Eliminates special characters/spaces
    - Disambiguates characters based on plate structure
    """
    # Remove all non-alphanumeric characters except space
    cleaned = re.sub(r'[^A-Za-z0-9 ]', '', raw_text).strip().upper()
    tokens = cleaned.split()

    fixed_tokens = []
    for token in tokens:
        # If token has 'J' immediately preceding digits (e.g. 'VJ16' where '0' was misread as 'J'),
        # or pattern like 'V J 16' / 'VO16', standardize 'VJ' followed by digits to 'V0' / 'VO'
        t = token
        # Replace 'J' between a letter and digits (e.g., 'VJ16' -> 'V016')
        t = re.sub(r'(^[A-Z])J(\d)', r'\g<1>0\2', t)
        # If standalone 'J' in between letters and numbers
        if t == 'J':
            t = '0'
        fixed_tokens.append(t)

    return " ".join(fixed_tokens)

def preprocess_and_read_plate(crop, reader):
    if crop is None or crop.size == 0:
        return "", 0.0

    passes = enhance_plate_for_ocr(crop)

    best_text = ""
    best_conf = 0.0

    for name, img_pass in passes:
        results = reader.readtext(
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
            raw_text = " ".join(text_parts).upper()
            processed_text = clean_plate_text(raw_text)
            avg_conf = float(np.mean(conf_parts))

            if avg_conf > best_conf and len(re.sub(r'[^A-Za-z0-9]', '', processed_text)) >= 2:
                best_conf = avg_conf
                best_text = processed_text

        if best_conf > 0.65 and len(re.sub(r'[^A-Za-z0-9]', '', best_text)) >= 4:
            break

    return best_text, best_conf

def test_pipeline():
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    for img_name in sorted(os.listdir('test_images')):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')): continue
        img = cv2.imread(os.path.join('test_images', img_name))
        text, conf = preprocess_and_read_plate(img, reader)
        print(f"[{img_name}] Output: '{text}' (conf: {conf:.2f})")

if __name__ == "__main__":
    test_pipeline()
