import json
import os
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm

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

def main():
    script_dir = Path(__file__).resolve().parent

    # Paths to your data
    documents_path = os.path.join(script_dir, "../../../data/raw/documents")
    pending_docs_path = os.path.join(script_dir, "../../../data/processed/pending_documents.json")
    filtered_path = os.path.join(script_dir, "../../../data/processed/filtered.json")
    filtered_with_text_v3_path = os.path.join(script_dir, "../../../data/processed/filtered_with_text-v3.json")

    # 1. Load pending documents (they should have at least an "id" field)
    with open(pending_docs_path, "r", encoding="utf-8") as fp:
        pending_docs = json.load(fp)

    # 2. Load existing dataset from filtered.json
    with open(filtered_path, "r", encoding="utf-8") as fp:
        records = json.load(fp)

    failures = []
    new_records = []

    # 3. Iterate over each pending document
    for doc_item in tqdm(pending_docs, desc="Processing pending documents"):
        doc_id = doc_item["id"]

        # Only check for .pdf extension
        pdf_path = os.path.join(documents_path, f"{doc_id}.pdf")
        if not os.path.exists(pdf_path):
            failures.append(doc_id)
            continue

        # Extract text via OCR
        text = extract_text_ocr(pdf_path)
        if not text.strip():
            failures.append(doc_id)
            continue

        # Create a new record in the same structure as your existing JSON
        new_record = {
            "documents": [
                {
                    "id": doc_id,
                    "url": doc_item.get("url", ""),  # or whatever you need
                    "documentText": text
                }
            ]
        }
        new_records.append(new_record)

    # 4. Merge new records into the existing dataset
    records.extend(new_records)

    # 5. Save combined data to filtered_with_text-v3.json
    with open(filtered_with_text_v3_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)

    # 6. Print any failures
    if failures:
        print("\nThe following document IDs failed to process or had no text:")
        for fail_id in failures:
            print(f" - {fail_id}")

if __name__ == "__main__":
    main()
