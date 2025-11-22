# PAN Card Extractor - AI-Powered OCR Application

A Streamlit-based web application that extracts information from Indian PAN (Permanent Account Number) cards using OCR (Optical Character Recognition) and OpenAI's GPT-4 API.

## Features

- 📄 Supports multiple file formats: PDF, JPG, JPEG, PNG, TIF, TIFF
- 🤖 AI-powered extraction using OpenAI GPT-4o-mini
- 🔍 Advanced image preprocessing for better OCR accuracy
- 📊 Exports results to CSV format
- 🎯 Extracts: PAN Number, Name, Father's Name, Date of Birth
- 🔄 Fallback mechanisms for robust extraction

## Prerequisites

- Python 3.8+
- Tesseract OCR installed on your system
- OpenAI API key

## Installation

1. **Clone or download this repository**

2. **Install Tesseract OCR**
   - Windows: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment**
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

5. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

6. **Create a `.env` file in the project directory**
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`

3. **Upload PAN card images or PDFs**
   - Click "Browse files" or drag and drop
   - Select one or multiple files

4. **View extracted data**
   - Results appear in a table format
   - Download as CSV using the "Download results as CSV" button

---

## Code Explanation - Line by Line

### Imports Section (Lines 1-15)

```python
import os
```
- Imports the `os` module for operating system operations (used for environment variables)

```python
import io
```
- Imports `io` for handling byte streams (used when processing uploaded files)

```python
import re
```
- Imports regular expressions module for pattern matching (PAN numbers, dates)

```python
import json
```
- Imports JSON module for parsing OpenAI API responses

```python
from typing import List, Dict, Tuple
```
- Imports type hints for better code documentation and IDE support

```python
import streamlit as st
```
- Imports Streamlit framework for creating the web interface

```python
from PIL import Image, ImageOps
```
- Imports Python Imaging Library for image manipulation
- `Image`: Basic image handling
- `ImageOps`: Image operations like grayscale conversion

```python
import fitz  # PyMuPDF
```
- Imports PyMuPDF library for PDF processing and rendering

```python
import pytesseract
```
- Imports Python wrapper for Tesseract OCR engine

```python
import cv2
```
- Imports OpenCV for advanced image preprocessing

```python
import numpy as np
```
- Imports NumPy for numerical array operations (used with images)

```python
import pandas as pd
```
- Imports Pandas for data manipulation and CSV export

```python
from openai import OpenAI
```
- Imports OpenAI client for API communication

```python
from dotenv import load_dotenv
```
- Imports function to load environment variables from `.env` file

---

### Configuration Section (Lines 17-24)

```python
load_dotenv()
```
- Loads environment variables from `.env` file (like OPENAI_API_KEY)

```python
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```
- Creates OpenAI client instance using API key from environment variable
- `os.getenv()` retrieves the API key securely

```python
PAN_REGEX = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", re.IGNORECASE)
```
- Compiles regex pattern to match PAN card format: 5 letters + 4 digits + 1 letter
- `\b` = word boundary
- `[A-Z]{5}` = exactly 5 letters
- `[0-9]{4}` = exactly 4 digits
- `[A-Z]` = exactly 1 letter
- `re.IGNORECASE` = case-insensitive matching

```python
DOB_REGEX = re.compile(r"\b([0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\b")
```
- Compiles regex pattern for date format: DD/MM/YYYY or DD-MM-YYYY
- Matches 2 digits, separator (/ or -), 2 digits, separator, 4 digits

---

### Image Preprocessing Function (Lines 28-38)

```python
def preprocess_image_pil(pil_image: Image.Image, upscale: int = 4) -> Image.Image:
```
- Defines function to preprocess images for better OCR accuracy
- Takes PIL Image and upscale factor (default 4x)

```python
    image = pil_image.convert("RGB")
```
- Converts image to RGB color mode (ensures consistent format)

```python
    w, h = image.size
```
- Gets current width and height of the image

```python
    image = image.resize((w * upscale, h * upscale), Image.LANCZOS)
```
- Upscales image 4x using high-quality LANCZOS resampling
- Larger images = better OCR accuracy

```python
    gray = ImageOps.grayscale(image)
```
- Converts image to grayscale (OCR works better on grayscale)

```python
    arr = np.array(gray)
```
- Converts PIL Image to NumPy array for OpenCV processing

```python
    arr = cv2.bilateralFilter(arr, 9, 75, 75)
```
- Applies bilateral filter to reduce noise while preserving edges
- Parameters: kernel size 9, color sigma 75, space sigma 75

```python
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```
- Applies Otsu's automatic threshold to create binary (black & white) image
- Improves text clarity by making text pure black on white background

```python
    return Image.fromarray(arr)
```
- Converts NumPy array back to PIL Image and returns it

---

### Multi-PSM OCR Function (Lines 41-49)

```python
def extract_text_multi_psm(pil_image: Image.Image) -> str:
```
- Defines function to extract text using multiple Tesseract PSM modes

```python
    text_all = []
    psms = [6, 7, 11]
```
- Creates empty list for text results
- PSM modes: 6 (uniform text block), 7 (single line), 11 (sparse text)

```python
    for psm in psms:
        config = f"--oem 3 --psm {psm}"
```
- Loops through each PSM mode
- `--oem 3` = use LSTM neural network OCR engine
- `--psm` = page segmentation mode

```python
        text = pytesseract.image_to_string(pil_image, config=config)
        text_all.append(text)
```
- Runs Tesseract OCR with current configuration
- Appends extracted text to list

```python
    return "\n".join(text_all)
```
- Joins all extracted text with newlines and returns combined result

---

### PDF to Images Converter (Lines 52-61)

```python
def render_pdf_to_images(pdf_bytes: bytes, zoom: int = 3) -> List[Image.Image]:
```
- Defines function to convert PDF pages to images
- Takes PDF as bytes, zoom level (default 3x for clarity)

```python
    images: List[Image.Image] = []
```
- Creates empty list to store converted images

```python
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
```
- Opens PDF from byte stream using PyMuPDF
- `with` ensures proper cleanup

```python
        for page in doc:
```
- Iterates through each page in the PDF

```python
            mat = fitz.Matrix(zoom, zoom)
```
- Creates transformation matrix for zooming (3x scale)

```python
            pix = page.get_pixmap(matrix=mat, alpha=False)
```
- Renders page to pixmap (raster image) with zoom applied
- `alpha=False` = no transparency channel

```python
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
```
- Converts pixmap to PIL Image in RGB mode

```python
            images.append(img)
```
- Adds converted image to list

```python
    return images
```
- Returns list of all page images

---

### PAN Regex Extraction (Lines 65-72)

```python
def extract_pan_regex(text: str) -> str:
```
- Defines function to extract PAN number using regex

```python
    match = PAN_REGEX.search(text)
```
- Searches for PAN pattern in text using compiled regex

```python
    if match:
        pan = match.group(1).upper()
```
- If match found, extracts first capture group and converts to uppercase

```python
        pan = pan.replace("0", "O").replace("1", "I").replace("5", "S")
```
- Corrects common OCR mistakes:
  - Zero (0) → Letter O
  - One (1) → Letter I
  - Five (5) → Letter S

```python
        return pan
    return ""
```
- Returns corrected PAN or empty string if not found

---

### DOB Regex Extraction (Lines 75-80)

```python
def extract_dob_regex(text: str) -> str:
```
- Defines function to extract date of birth using regex

```python
    match = DOB_REGEX.search(text)
    if match:
        return match.group(1)
    return ""
```
- Searches for date pattern, returns matched date or empty string

---

### Fallback Name Extraction (Lines 83-136)

```python
def extract_name_and_father_fallback(text: str) -> Tuple[str, str]:
```
- Defines fallback function to extract names when AI extraction fails
- Returns tuple of (name, father_name)

```python
    lines = [line.strip() for line in text.split("\n") if line.strip()]
```
- Splits text into lines, removes whitespace, filters out empty lines

```python
    exclude_keywords = [
        "INCOME", "GOVT", "INDIA", "TAX", "PERMANENT", "ACCOUNT",
        "NUMBER", "CARD", "DATE", "BIRTH", "SIGNATURE", "DEPARTMENT"
    ]
```
- Lists keywords to exclude (common PAN card headers, not names)

```python
    name = ""
    father_name = ""
```
- Initializes empty strings for results

```python
    for i, line in enumerate(lines):
        line_upper = line.upper()
```
- Loops through lines with index, converts to uppercase for comparison

```python
        if any(keyword in line_upper for keyword in ["NAME", "नाम"]):
```
- Checks if line contains "Name" keyword (English or Hindi)

```python
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
```
- Gets the next line (actual name usually appears after the label)

```python
                if (next_line.isupper() and 
                    1 <= len(next_line.split()) <= 4 and 
                    not any(char.isdigit() for char in next_line) and
                    not any(kw in next_line for kw in exclude_keywords)):
                    name = next_line
```
- Validates the next line as a name:
  - Must be all uppercase
  - Must have 1-4 words
  - Must not contain digits
  - Must not contain excluded keywords
- If valid, assigns as name

```python
        if any(keyword in line_upper for keyword in ["FATHER", "पिता"]):
```
- Same logic for "Father" keyword (English or Hindi)

```python
    if not name or not father_name:
        valid_lines = []
        for line in lines:
            if (line.isupper() and 
                2 <= len(line.split()) <= 4 and
                not any(char.isdigit() for char in line) and
                not any(kw in line for kw in exclude_keywords) and
                re.match(r'^[A-Z ]+$', line)):
                valid_lines.append(line)
```
- If names not found by keywords, creates list of all valid uppercase lines
- Valid lines: uppercase, 2-4 words, no digits, no keywords, only letters and spaces

```python
        if not name and len(valid_lines) > 0:
            name = valid_lines[0]
        if not father_name and len(valid_lines) > 1:
            father_name = valid_lines[1]
```
- Uses first valid line as name, second as father's name (common PAN card layout)

```python
    return name, father_name
```
- Returns tuple of extracted names

---

### OpenAI Extraction Function (Lines 140-213)

```python
def extract_pan_fields_with_openai(text: str) -> Dict:
```
- Main extraction function using OpenAI GPT-4o-mini

```python
    prompt = f"""
You are an OCR post-processor for Indian PAN Cards.
...
"""
```
- Creates detailed prompt with instructions for AI:
  - Specifies exact format for PAN, names, DOB
  - Provides validation rules
  - Requests JSON output format

```python
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400
        )
```
- Calls OpenAI API with:
  - Model: gpt-4o-mini (fast, cost-effective)
  - Temperature: 0 (deterministic output)
  - Max tokens: 400 (enough for JSON response)

```python
        content = response.choices[0].message.content.strip()
```
- Extracts text content from API response

```python
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
```
- Handles cases where AI returns JSON wrapped in code blocks
- Extracts just the JSON content

```python
        result = json.loads(content)
```
- Parses JSON string into Python dictionary

```python
        if "pan_number" in result:
            result["pan_number"] = result["pan_number"].strip().upper()
        if "name" in result:
            result["name"] = result["name"].strip().upper()
        if "father_name" in result:
            result["father_name"] = result["father_name"].strip().upper()
        if "dob" in result:
            result["dob"] = result["dob"].strip()
```
- Validates and cleans each field:
  - Removes extra whitespace
  - Converts to uppercase (except DOB)

```python
        return result
    except Exception as e:
        return {"pan_number": "", "name": "", "father_name": "", "dob": ""}
```
- Returns cleaned results or empty dictionary if error occurs

---

### Streamlit UI Setup (Lines 217-221)

```python
st.set_page_config(page_title="PAN Extractor — OpenAI", layout="wide")
```
- Configures Streamlit page with title and wide layout

```python
st.title("PAN Extractor ")
```
- Displays main title at top of page

```python
uploaded_files = st.file_uploader(
    "Upload files", accept_multiple_files=True,
    type=["pdf", "jpg", "jpeg", "png", "tif", "tiff"]
)
```
- Creates file uploader widget
- Allows multiple file selection
- Restricts to specified file types

```python
results_rows: List[Dict] = []
```
- Initializes empty list to store extraction results

---

### File Processing Loop (Lines 224-285)

```python
if uploaded_files:
    total = len(uploaded_files)
    for idx, file in enumerate(uploaded_files, start=1):
```
- Checks if files uploaded
- Counts total files
- Loops through each file with index starting at 1

```python
        st.info(f"Processing {file.name} ({idx}/{total})")
```
- Displays info message showing current progress

```python
        fname = file.name
        data = file.read()
```
- Stores filename and reads file content as bytes

```python
        if fname.lower().endswith('.pdf'):
            try:
                images = render_pdf_to_images(data, zoom=3)
            except Exception as e:
                st.error(f"Failed PDF {fname}: {e}")
                continue
```
- Checks if file is PDF
- Converts PDF to images with 3x zoom
- Shows error and continues to next file if conversion fails

```python
        else:
            try:
                images = [Image.open(io.BytesIO(data))]
            except Exception as e:
                st.error(f"Failed image {fname}: {e}")
                continue
```
- For non-PDF files, opens as single image
- Wraps in list for consistent processing

```python
        found = False
        for page_no, img in enumerate(images, start=1):
```
- Initializes flag to track if data found
- Loops through each image/page

```python
            pre = preprocess_image_pil(img, upscale=4)
```
- Preprocesses image with 4x upscaling for better OCR

```python
            text = extract_text_multi_psm(pre)
```
- Extracts text using multiple Tesseract PSM modes

```python
            with st.expander(f"OCR Output Preview — {fname}, Page {page_no}"):
                st.text_area("OCR Text", text, height=200)
```
- Creates collapsible section showing raw OCR text
- Useful for debugging

```python
            pan = extract_pan_regex(text)
            dob = extract_dob_regex(text)
```
- Extracts PAN and DOB using regex fallback methods

```python
            extracted = extract_pan_fields_with_openai(text)
```
- Uses OpenAI to extract all fields intelligently

```python
            if pan and not extracted.get("pan_number"):
                extracted["pan_number"] = pan
            if dob and not extracted.get("dob"):
                extracted["dob"] = dob
```
- If regex found values that AI missed, fills them in

```python
            if not extracted.get("name") or not extracted.get("father_name"):
                fallback_name, fallback_father = extract_name_and_father_fallback(text)
                if not extracted.get("name") and fallback_name:
                    extracted["name"] = fallback_name
                if not extracted.get("father_name") and fallback_father:
                    extracted["father_name"] = fallback_father
```
- If AI missed names, tries pattern-based fallback extraction

```python
            if any(extracted.values()):
                found = True
                results_rows.append({
                    "file": fname,
                    "page": page_no,
                    "pan_number": extracted.get("pan_number", ""),
                    "name": extracted.get("name", ""),
                    "father_name": extracted.get("father_name", ""),
                    "dob": extracted.get("dob", "")
                })
```
- If any field extracted successfully, marks as found
- Appends result dictionary to results list

```python
        if not found:
            results_rows.append({
                "file": fname,
                "page": "-",
                "pan_number": "",
                "name": "",
                "father_name": "",
                "dob": ""
            })
```
- If no data found in any page, adds empty row with filename

---

### Results Display (Lines 287-301)

```python
if results_rows:
    df = pd.DataFrame(results_rows)
```
- If results exist, converts list of dictionaries to Pandas DataFrame

```python
    st.dataframe(df)
```
- Displays DataFrame as interactive table in Streamlit

```python
    csv = df.to_csv(index=False).encode("utf-8")
```
- Converts DataFrame to CSV format
- `index=False` excludes row numbers
- Encodes as UTF-8 bytes

```python
    st.download_button(
        "Download results as CSV",
        data=csv,
        file_name="pan_extraction_results.csv",
        mime="text/csv"
    )
```
- Creates download button for CSV file
- Provides default filename and MIME type

```python
else:
    if uploaded_files:
        st.warning("No PAN details detected.")
    else:
        st.write("Upload files to start extraction.")
```
- Shows appropriate message:
  - Warning if files uploaded but nothing extracted
  - Instructions if no files uploaded yet

---

## Architecture & Flow

1. **File Upload** → User uploads PDF/image files
2. **Preprocessing** → Images upscaled, converted to grayscale, filtered, binarized
3. **OCR** → Tesseract extracts text using multiple PSM modes
4. **AI Extraction** → OpenAI GPT-4o-mini intelligently parses OCR text
5. **Fallback** → Regex and pattern matching fill gaps if AI misses data
6. **Display** → Results shown in table with download option

## Technologies Used

- **Streamlit**: Web interface
- **Tesseract OCR**: Text extraction from images
- **OpenCV**: Image preprocessing
- **PyMuPDF**: PDF rendering
- **OpenAI GPT-4o-mini**: Intelligent field extraction
- **Pillow**: Image manipulation
- **Pandas**: Data handling and export

## Error Handling

- PDF conversion failures are caught and logged
- OpenAI API errors return empty results gracefully
- Invalid images are skipped with error messages
- Missing data fields return empty strings instead of crashing

## Performance Optimization

- Multiple PSM modes for better OCR accuracy
- 4x image upscaling for clearer text
- Bilateral filtering for noise reduction
- Cascading fallback mechanisms (AI → Regex → Pattern matching)

## License

This project is for educational purposes. Ensure compliance with data privacy regulations when processing PAN cards.

## Support

For issues or questions, please check:
- Tesseract is properly installed
- OpenAI API key is valid and has credits
- Input files are clear and readable

---

**Created for extracting PAN card information efficiently and accurately using modern AI and OCR technologies.**
