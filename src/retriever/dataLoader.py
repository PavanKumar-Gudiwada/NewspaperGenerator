import os
import pandas as pd
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract

def load_all_data(data_folder = "data"):
    """
    Load and combine text data from structured (CSV, Excel) and unstructured (PDF, images) files.
    """
    
    structured_folder = os.path.join(data_folder, "structured")
    unstructured_folder = os.path.join(data_folder, "un_structured")

    all_texts = []

    # --- Load structured data ---
    for file in os.listdir(structured_folder):
        file_path = os.path.join(structured_folder, file)
        if file.endswith(".csv"):
            df = pd.read_csv(file_path)
            all_texts.append(df.to_string())
        elif file.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
            all_texts.append(df.to_string())

    # --- Load unstructured data ---
    for file in os.listdir(unstructured_folder):
        file_path = os.path.join(unstructured_folder, file)
        if file.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            all_texts.append(text)
        elif file.endswith((".png", ".jpeg", ".jpg")):
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            all_texts.append(text)

    # --- Combine all text ---
    combined_text = "\n".join(all_texts)
    print("Loaded text length:", len(combined_text))

    return combined_text
