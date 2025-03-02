import json
from pathlib import Path
import os
import requests
from tqdm import tqdm
from mimetypes import guess_extension

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent

filtered_dataset_path = os.path.join(script_dir, r"../../../data/processed/filtered.json")
download_path = os.path.join(script_dir, r"../../../data/raw/documents")

def download_file(url, name):
    """Downloads file from a given URL"""

    response = requests.get(url)
    
    responseType = response.headers.get("Content-Type", None) 
    ext = guess_extension(responseType)

    file_path = os.path.join(download_path, f"{name}{ext}")
    with open(file_path, 'wb') as fp:
        fp.write(response.content)


with open(filtered_dataset_path, "r", encoding="utf-8") as fp:
    records = json.load(fp)

# Iterate over documents
for record in tqdm(records):
    for document in record["documents"]:
        # Download file
        url = document["url"]
        download_file(url, f'{document["id"]}')
