import json
from pathlib import Path
import os
import fitz  # PyMuPDF
from docx import Document
from tqdm import tqdm

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent

documents_path = os.path.join(script_dir, r"../../../data/raw/")
dataset_path = os.path.join(script_dir, r"../../../data/processed/filtered.json")
dataset_with_text_path = os.path.join(script_dir, r"../../../data/processed/filtered_with_text.json")

def extract_text(documentId):
    try:
        try:
            path = os.path.join(documents_path, f"{documentId}.pdf")
            document = fitz.open(path)
            text = "\n".join([page.get_text() for page in document])
            return text
        except:
            path = os.path.join(documents_path, f"{documentId}.doc")
            doc = Document(path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
    except:
        return ""

with open(dataset_path, "r", encoding="utf-8") as fp:
    records = json.load(fp)

# Iterate over documents
for record in tqdm(records):
    for document in record["documents"]:
        # Extract text
        url = document["url"]
        document["documentText"] = extract_text(document["id"])
    
# Save dataset with texts
with open(dataset_with_text_path, "w") as fp:
    json.dump(records, fp)
