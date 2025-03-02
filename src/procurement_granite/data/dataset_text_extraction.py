import json
from pathlib import Path
import os
import fitz  # PyMuPDF
from docx import Document
from tqdm import tqdm
import pytesseract
from pdf2image import convert_from_path

# Set environment variables
os.environ["PATH"] += ";C:\\poppler\\Library\\bin"
os.environ["PATH"] += ";C:\\Program Files\\Tesseract-OCR"

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent

documents_path = "D:\projects\procurement-llm\documents"
filtered_dataset_path = os.path.join(script_dir, r"../../../data/processed/filtered.json")
full_dataset_path = os.path.join(script_dir, r"../../../data/processed/filtered_with_text.json")

def extract_text_ocr(pdf_file: str) -> str:
    """
    Converts each page of a PDF to an image and uses Tesseract OCR
    to extract text. Useful for scanned PDFs with no embedded text layer.
    """
    pages = convert_from_path(pdf_file)
    extracted_text = ""
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        extracted_text += f"--- Page {i+1} ---\n{text}\n"
    return extracted_text

def extract_text(document_path):
    document_type = document_path.split("/")[-1].split(".")[1]
    text = ""

    if document_type == "pdf":
        document = fitz.open(document_path)
        text = "\n".join([page.get_text() for page in document])
        text = text.strip()

        if len(text) <= 30:
            text = extract_text_ocr(document_path)

    elif document_type == "doc":
        try:
            doc = Document(document_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            text = text.strip()
        except:
            text = ""
    
    return text

# Iterate over documents
id_to_text = {}

paths = [os.path.join(documents_path, file) for file in os.listdir(documents_path)]
for document_path in tqdm(paths):
    # Extract text
    document_id = document_path.split("/")[-1].split(".")[0]
    id_to_text[document_id] = extract_text(document_path)


with open(filtered_dataset_path, "r", encoding="utf-8") as fp:
    records = json.load(fp)

for record in records:
    for document in record["documents"]:
        document["documentText"] = id_to_text[document["id"]]

# Save full dataset
print(len(records))
with open(full_dataset_path, "w") as fp:
    json.dump(records, fp)