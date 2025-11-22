import os
import io
import re
import json
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image, ImageOps
import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Regex patterns
PAN_REGEX = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", re.IGNORECASE)
DOB_REGEX = re.compile(r"\b([0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\b")


# ---------------------- OCR HELPERS ---------------------- #
def preprocess_image_pil(pil_image: Image.Image, upscale: int = 4) -> Image.Image:
    """Preprocess image for better OCR results."""
    image = pil_image.convert("RGB")
    w, h = image.size
    image = image.resize((w * upscale, h * upscale), Image.LANCZOS)
    gray = ImageOps.grayscale(image)
    arr = np.array(gray)
    arr = cv2.bilateralFilter(arr, 9, 75, 75)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(arr)


def extract_text_multi_psm(pil_image: Image.Image) -> str:
    """Try multiple PSM modes in Tesseract to improve OCR accuracy."""
    text_all = []
    psms = [6, 7, 11]
    for psm in psms:
        config = f"--oem 3 --psm {psm}"
        text = pytesseract.image_to_string(pil_image, config=config)
        text_all.append(text)
    return "\n".join(text_all)


def render_pdf_to_images(pdf_bytes: bytes, zoom: int = 3) -> List[Image.Image]:
    """Convert PDF pages to images (higher zoom for clarity)."""
    images: List[Image.Image] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


# ---------------------- FALLBACK HELPERS ---------------------- #
def extract_pan_regex(text: str) -> str:
    """Extract PAN number using regex + correction."""
    match = PAN_REGEX.search(text)
    if match:
        pan = match.group(1).upper()
        pan = pan.replace("0", "O").replace("1", "I").replace("5", "S")
        return pan
    return ""


def extract_dob_regex(text: str) -> str:
    """Extract DOB using regex."""
    match = DOB_REGEX.search(text)
    if match:
        return match.group(1)
    return ""


def extract_name_and_father_fallback(text: str) -> Tuple[str, str]:
    """Extract name and father's name using pattern matching."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    exclude_keywords = [
        "INCOME", "GOVT", "INDIA", "TAX", "PERMANENT", "ACCOUNT",
        "NUMBER", "CARD", "DATE", "BIRTH", "SIGNATURE", "DEPARTMENT"
    ]
    
    name = ""
    father_name = ""
    
    # Look for patterns
    for i, line in enumerate(lines):
        line_upper = line.upper()
        
        # Look for "Name" or similar keywords
        if any(keyword in line_upper for keyword in ["NAME", "नाम"]):
            # Check if name is on same line or next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # If next line is uppercase with 2-4 words and no numbers
                if (next_line.isupper() and 
                    1 <= len(next_line.split()) <= 4 and 
                    not any(char.isdigit() for char in next_line) and
                    not any(kw in next_line for kw in exclude_keywords)):
                    name = next_line
        
        # Look for "Father" or similar keywords
        if any(keyword in line_upper for keyword in ["FATHER", "पिता"]):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (next_line.isupper() and 
                    1 <= len(next_line.split()) <= 4 and 
                    not any(char.isdigit() for char in next_line) and
                    not any(kw in next_line for kw in exclude_keywords)):
                    father_name = next_line
    
    # If still not found, look for uppercase lines
    if not name or not father_name:
        valid_lines = []
        for line in lines:
            if (line.isupper() and 
                2 <= len(line.split()) <= 4 and
                not any(char.isdigit() for char in line) and
                not any(kw in line for kw in exclude_keywords) and
                re.match(r'^[A-Z ]+$', line)):
                valid_lines.append(line)
        
        if not name and len(valid_lines) > 0:
            name = valid_lines[0]
        if not father_name and len(valid_lines) > 1:
            father_name = valid_lines[1]
    
    return name, father_name


# ---------------------- OPENAI EXTRACTION ---------------------- #
def extract_pan_fields_with_openai(text: str) -> Dict:
    """
    Use OpenAI to extract PAN, Name, Father’s Name, and DOB from OCR text.
    """
    prompt = f"""
You are an OCR post-processor for Indian PAN Cards.
Given the raw OCR text, extract the following fields strictly:

- pan_number: 10 characters in format ABCDE1234F (5 letters + 4 digits + 1 letter)
- name: Card holder's full name (usually appears after "Name" field, in UPPERCASE)
- father_name: Father's full name (usually appears after "Father's Name" or "Father" field, in UPPERCASE)
- dob: Date of birth in DD/MM/YYYY format

IMPORTANT RULES:
1. Names should contain only alphabets and spaces, no numbers or special characters
2. Names are typically in UPPERCASE on PAN cards
3. Extract the actual name values, NOT field labels like "Name" or "Father's Name"
4. PAN number must match the exact format XXXXX9999X
5. Look carefully at the text structure - names usually appear as separate lines after their labels

Return ONLY valid JSON with keys: pan_number, name, father_name, dob.
If a field cannot be extracted with confidence, return empty string for that field.

OCR text:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        # Validate and clean the results
        if "pan_number" in result:
            result["pan_number"] = result["pan_number"].strip().upper()
        if "name" in result:
            result["name"] = result["name"].strip().upper()
        if "father_name" in result:
            result["father_name"] = result["father_name"].strip().upper()
        if "dob" in result:
            result["dob"] = result["dob"].strip()
            
        return result
    except Exception as e:
        return {"pan_number": "", "name": "", "father_name": "", "dob": ""}


# ---------------------- STREAMLIT UI ---------------------- #
st.set_page_config(page_title="PAN Extractor — OpenAI", layout="wide")
st.title("PAN Extractor ")

uploaded_files = st.file_uploader(
    "Upload files", accept_multiple_files=True,
    type=["pdf", "jpg", "jpeg", "png", "tif", "tiff"]
)

results_rows: List[Dict] = []

if uploaded_files:
    total = len(uploaded_files)
    for idx, file in enumerate(uploaded_files, start=1):
        st.info(f"Processing {file.name} ({idx}/{total})")
        fname = file.name
        data = file.read()

        # Convert to images
        if fname.lower().endswith('.pdf'):
            try:
                images = render_pdf_to_images(data, zoom=3)
            except Exception as e:
                st.error(f"Failed PDF {fname}: {e}")
                continue
        else:
            try:
                images = [Image.open(io.BytesIO(data))]
            except Exception as e:
                st.error(f"Failed image {fname}: {e}")
                continue

        found = False
        for page_no, img in enumerate(images, start=1):
            pre = preprocess_image_pil(img, upscale=4)
            text = extract_text_multi_psm(pre)

            # Show OCR preview for debugging
            with st.expander(f"OCR Output Preview — {fname}, Page {page_no}"):
                st.text_area("OCR Text", text, height=200)

            # Regex fallback
            pan = extract_pan_regex(text)
            dob = extract_dob_regex(text)

            # OpenAI extraction
            extracted = extract_pan_fields_with_openai(text)

            # If regex found something missing in OpenAI output, fill it
            if pan and not extracted.get("pan_number"):
                extracted["pan_number"] = pan
            if dob and not extracted.get("dob"):
                extracted["dob"] = dob
            
            # If names are missing, try fallback extraction
            if not extracted.get("name") or not extracted.get("father_name"):
                fallback_name, fallback_father = extract_name_and_father_fallback(text)
                if not extracted.get("name") and fallback_name:
                    extracted["name"] = fallback_name
                if not extracted.get("father_name") and fallback_father:
                    extracted["father_name"] = fallback_father

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

        if not found:
            results_rows.append({
                "file": fname,
                "page": "-",
                "pan_number": "",
                "name": "",
                "father_name": "",
                "dob": ""
            })

if results_rows:
    df = pd.DataFrame(results_rows)
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv,
        file_name="pan_extraction_results.csv",
        mime="text/csv"
    )
else:
    if uploaded_files:
        st.warning("No PAN details detected.")
    else:
        st.write("Upload files to start extraction.")
