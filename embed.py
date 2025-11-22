import time
import uuid
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

PROGRESS_FILE = "progress.json"

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
    text_parts = [soup.get_text(" ", strip=True)]
    for img in soup.find_all("img"):
        if img.has_attr("src"):
            text_parts.append(f"[Image: {img['src']}]")
    return " ".join(p for p in text_parts if p)

def clean_row(row: dict) -> dict:
    """Clean all fields: strip HTML if present"""
    cleaned = {}
    for k, v in row.items():
        if isinstance(v, str):
            cleaned[k] = clean_html(v)
        else:
            cleaned[k] = v
    return cleaned

def load_progress():
    """Load last processed row and file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"file": None, "row": -1}  # -1 means none processed yet

def save_progress(filename, row_num):
    """Save progress for next resume"""
    progress = {"file": filename, "row": row_num}
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)
    print(f"💾 Progress saved: {progress}")

def embed_text(text: str):
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="SEMANTIC_SIMILARITY"
    )
    return np.array(response["embedding"])

def main():
    # Load environment variables
    load_dotenv()

    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    # Load dataset
    filename = "products_data.csv"
    df = pd.read_csv(filename, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Progress
    progress = load_progress()
    start_row = progress.get("row", -1) + 1

    # Init Qdrant
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )

    sku_col = "sku" if "sku" in df.columns else None

    for idx, row in df.iterrows():
        if idx < start_row:
            continue

        cleaned_row = clean_row(row)
        product_text = " ".join([str(value) for value in cleaned_row.values()])
        embedding = embed_text(product_text)

        if sku_col:
            point_id = sku_to_id(row[sku_col])
        else:
            point_id = idx

        payload = {
            "content": product_text,
            "metadata": {col: row[col] for col in df.columns}
        }

        qdrant_client.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=point_id, vector=embedding.tolist(), payload=payload)]
        )

        save_progress(filename, idx)

    print("✅ All rows processed successfully.")

def safe_main():
    while True:
        try:
            print("🚀 Starting main process...")
            main()
            print("🎉 Main process completed successfully.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}. Retrying in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    safe_main()
