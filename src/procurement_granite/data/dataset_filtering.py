import json
from pathlib import Path
import os

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent

raw_path = os.path.join(script_dir, r"../../../data/raw/raw.jsonl")
filtered_path = os.path.join(script_dir, r"../../../data/processed/filtered.json")

filtered = []

# Keywords for filtering procurements related to connectivity
keywords = [
    "connectivity", "network", "telecommunications", "internet", "broadband",
    "wireless", "fiber", "ethernet", "satellite", "5g"
]

# Function to check if text contains any keywords
def contains_keywords(text, keywords):
    text = text.lower()
    return any(keyword in text for keyword in keywords)

with open(raw_path, "r", encoding="utf-8") as fp:
    for line in fp:

        # Load record as dict
        record = json.loads(line)

        # Filter by keywords
        title = record.get("tender", {}).get("title", "").lower()
        description = record.get("tender", {}).get("description", "").lower()


        eligibilityCriteria = record.get("tender", {}).get("eligibilityCriteria", None)
        if eligibilityCriteria is None:
            continue

        if contains_keywords(title, keywords) or contains_keywords(description, keywords):

            documents = record["tender"].get("documents", [])
            documents = [
                document 
                for document in documents
                if (
                    document["title"] != "Bid Securing Declaration Required" and
                    document["documentType"] == "biddingDocuments"
                )
            ]

            # Keep relevant information
            record = {
                "id": record["id"],
                "title": record["tender"]["title"],
                "description": record["tender"]["description"],
                "eligibilityCriteria": record["tender"].get("eligibilityCriteria", None),
                "awardCriteria": record["tender"].get("awardCriteria", None),
                "awardCriteriaDetails": record["tender"].get("awardCriteriaDetails", None),
                "documents": documents
            }
            filtered.append(record)

print(len(filtered))

# Save filtered dataset
with open(filtered_path, "w") as fp:
    json.dump(filtered, fp)
