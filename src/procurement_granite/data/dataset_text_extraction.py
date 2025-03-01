import json
from pathlib import Path
import os
import requests
import tempfile
import pdfplumber
import fitz  # PyMuPDF
from docx import Document


# Get the directory of the current script
script_dir = Path(__file__).resolve().parent


dataset_path = os.path.join(script_dir, r"../../../data/processed/filtered.json")
dataset_with_text_path = os.path.join(script_dir, r"../../../data/processed/filtered_with_text.json")


def download_file(url):
    """Downloads a file from the given URL and saves it to a temporary file."""
    response = requests.get(url, stream=True)

    # if response.status_code == 200:
    #     ext = url.split('.')[-1].lower()
    #     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    #     temp_file.write(response.content)
    #     temp_file.close()
    #     return temp_file.name
    # else:
    #     raise Exception(f"Failed to download file: {response.status_code}")


with open(dataset_path, "r", encoding="utf-8") as fp:
    records = json.load(fp)

# Iterate over documents
for record in records:
    for document in record["documents"]:
        # Extract text
        url = document["url"]
        download_file(url)
        document["text"] = "test"
    

# Save dataset with texts
with open(dataset_with_text_path, "w") as fp:
    json.dump(records, fp)
