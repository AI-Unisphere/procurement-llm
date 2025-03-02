import json
from pathlib import Path
import os
import requests
from tqdm import tqdm

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent

dataset_path = os.path.join(script_dir, r"../../../data/processed/filtered.json")
filtered_output_path = os.path.join(script_dir, r"../../../data/processed/filtered_documents.json")

# List to store filtered documents
filtered_documents = []

# Target content types
target_content_types = {'application/msword', 'application/pdf'}

def get_content_type(url):
    """Fetches the Content-Type from the response headers."""
    try:
        response = requests.head(url, timeout=5)
        return response.headers.get("Content-Type", "").split(";")[0]  # Remove encoding info
    except requests.RequestException as e:
        print(f"Error fetching content type for {url}: {e}")
        return None

# Load dataset
with open(dataset_path, "r", encoding="utf-8") as fp:
    records = json.load(fp)

# Iterate over documents
for record in tqdm(records, desc="Filtering Documents"):
    for document in record.get("documents", []):
        url = document.get("url")
        document_id = document.get("id")
        
        if url and document_id:
            content_type = get_content_type(url)
            if content_type in target_content_types:
                filtered_documents.append({
                    "id": document_id,
                    "url": url,
                    "content_type": content_type
                })

# Save filtered documents
with open(filtered_output_path, "w", encoding="utf-8") as fp:
    json.dump(filtered_documents, fp, indent=4)

print(f"Filtered documents saved to {filtered_output_path}")
