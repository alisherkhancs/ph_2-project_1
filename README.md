# 🚗 Vehicle Number Plate Recognition System

A simple, lightweight Computer Vision web application that detects vehicle number plates from images, crops the plate region, extracts the alphanumeric plate text using OCR, and displays the results in an interactive web interface.

---

## 📌 Project Workflow

```text
[ Vehicle Image Input ] 
          ⬇
[ Number Plate Detection ] (Haar Cascade / Contour Analysis)
          ⬇
[ Plate Cropping ] (Extracting Bounding Box)
          ⬇
[ OCR Processing ] (EasyOCR Text Extraction)
          ⬇
[ Result Display ] (Streamlit Web Interface)
```

---

## 🛠️ Technologies Used

* **Python 3.11** - Core programming language
* **OpenCV (`opencv-python`)** - Image reading, preprocessing, contour analysis, and Haar Cascade object detection
* **EasyOCR (`easyocr`)** - Optical Character Recognition for reading alphanumeric plate text
* **NumPy (`numpy`)** - Matrix and array manipulations for image processing
* **Streamlit (`streamlit`)** - Interactive web user interface
* **Haar Cascade XML** (`haarcascade_russian_plate_number.xml`) - Pre-trained license plate cascade classifier

---

## 🚀 How to Run the Application

### 1. Activate the Virtual Environment
Open PowerShell or Command Prompt in the project folder and run:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run the Streamlit Application
```powershell
streamlit run app.py
```
Or directly using the virtual environment executable:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### 3. Open the App in Your Browser
Navigate to:
```text
http://localhost:8501
```

---

## 📖 How to Use

1. Open the application at `http://localhost:8501`.
2. In the left sidebar:
   * Choose **"Use Sample Vehicle Image"** to quickly test the 5 pre-loaded sample vehicles (`car_1.jpg` to `car_5.jpg`).
   * OR choose **"Upload Custom Image"** to upload any `.jpg` or `.png` photo of a vehicle.
3. The system will automatically:
   * Display the original vehicle image.
   * Highlight the detected number plate with a green bounding box.
   * Display the cropped plate image.
   * Display the extracted license plate text in bold.

---

## 📁 Project Structure

```text
ph_2 project_1/
├── .venv/                                # Isolated Python virtual environment
├── test_images/                          # Sample vehicle test images
│   ├── car_1.jpg
│   ├── car_2.jpg
│   ├── car_3.jpg
│   ├── car_4.jpg
│   └── car_5.jpg
├── app.py                                # Main Streamlit web application
├── haarcascade_russian_plate_number.xml  # OpenCV Haar Cascade model for license plates
├── test_pipeline.py                      # Standalone CLI validation test script
└── README.md                             # Project documentation
```
