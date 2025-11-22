import os
import glob
import csv
import uuid
import json
import re
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# ---------------- CONFIG ----------------
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "oygula-products"
GEMINI_API_KEY = os.getenv("GEMINI_KEY")
PROGRESS_FILE = "progress.json"

# ----------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))

# Create collection if not exists (Gemini embeddings = 3072 dims)
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )

def embed_text(text: str):
    """Generate embeddings using Gemini"""
    resp = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text
    )
    return resp["embedding"]

def sku_to_id(sku: str):
    """Convert SKU to Qdrant ID (int if numeric, string UUID if alphanumeric)"""
    try:
        return int(float(sku))  # handles "541123785.0"
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, sku))  # convert to string

def clean_html(value: str) -> str:
    """Strip HTML tags but keep text + image URLs"""
    if not value or "<" not in value:
        return value

    soup = BeautifulSoup(value, "html.parser")

    # Extract text
    text_parts = [soup.get_text(" ", strip=True)]

    # Extract <img src="...">
    for img in soup.find_all("img"):
        if img.has_attr("src"):
            text_parts.append(f"[Image: {img['src']}]")

    return " ".join(p for p in text_parts if p)

def clean_row(row: dict) -> dict:
    """Clean all fields: strip HTML if present"""
    return {k: clean_html(v) if isinstance(v, str) else v for k, v in row.items()}

def load_progress():
    """Load last processed row and file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"file": None, "row": 0}

def save_progress(filename, row_num):
    """Save progress (row_num - 2)"""
    progress = {"file": filename, "row": max(0, row_num - 2)}
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)
    print(f"💾 Progress saved: {progress}")

def process_csv(file_path: str, start_row: int = 0):
    """Read CSV and push rows as embeddings with column mapping"""
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_sku = "sku" in reader.fieldnames

        for row_num, row in enumerate(reader, start=1):
            if row_num <= start_row:
                continue  # skip already processed

            # Clean HTML fields
            clean_fields = clean_row(row)

            # Build text for embedding (all columns concatenated)
            text = " | ".join(f"{k}: {v}" for k, v in clean_fields.items() if v)

            # Get embedding
            vector = embed_text(text)

            # Decide point ID
            if has_sku and row.get("sku"):
                point_id = sku_to_id(row["sku"])
            else:
                point_id = str(uuid.uuid4())  # random UUID

            # ✅ Store with `pageContent` (retriever expects this key)
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "pageContent": text,       # 👈 searchable text
                            "metadata": {              # 👈 structured fields
                                "source": file_path,
                                "row_num": row_num,
                                "fields": clean_fields
                            }
                        }
                    )
                ]
            )
            print(f"✅ Upserted ID: {point_id} (row {row_num})")

            # Save progress every row
            save_progress(file_path, row_num)

    print(f"🎯 Finished processing {file_path}")

def main():
    progress = load_progress()
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found in current directory.")
        return

    for csv_file in csv_files:
        start_row = progress["row"] if progress["file"] == csv_file else 0
        print(f"📂 Processing {csv_file} starting at row {start_row + 1}...")
        process_csv(csv_file, start_row=start_row)

    print("🎉 All CSV files pushed to Qdrant with structured fields.")

def safe_main():
    while True:
        try:
            print("🚀 Starting main function...")
            main()
            print("✅ Main function completed successfully.")
            break
        except Exception as e:
            print(f"❌ Script crashed with error: {e}")
            print("🔄 Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    safe_main()
