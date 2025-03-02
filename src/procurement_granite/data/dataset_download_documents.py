import json
from pathlib import Path
import os
import requests
from tqdm import tqdm
from mimetypes import guess_extension

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent

valid_files_path = os.path.join(script_dir, r"../../../data/processed/valid_files.json")
download_path = os.path.join(script_dir, r"../../../data/raw")

def download_file(url, name):
    """Downloads file from a given URL"""

    response = requests.get(url)
    
    responseType = response.headers.get("Content-Type", None) 
    ext = guess_extension(responseType)

    file_path = os.path.join(download_path, f"{name}{ext}")
    with open(file_path, 'wb') as fp:
        fp.write(response.content)


with open(valid_files_path, "r", encoding="utf-8") as fp:
    valid_files = json.load(fp)

# Iterate over documents
for valid_file in tqdm(valid_files[106:]):
    # Download file
    url = valid_file["url"]
    download_file(url, f'{valid_file["id"]}')
