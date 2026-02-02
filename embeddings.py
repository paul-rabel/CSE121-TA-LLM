import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------
# Paths (Adjust as needed)
# ------------------------------
RAW_JSON_PATH = "data/raw/cse121_dump_20260202_123416.json"
VECTOR_DIR = "data/vectors"
os.makedirs(VECTOR_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(VECTOR_DIR, "cse121_faiss.index")
METADATA_PATH = os.path.join(VECTOR_DIR, "metadata.json")

# ------------------------------
# 1. Load raw scraped content
# ------------------------------
if not os.path.exists(RAW_JSON_PATH):
    print(f"Error: {RAW_JSON_PATH} not found. Run your scraper first!")
    exit()

with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
    pages = json.load(f)

# ------------------------------
# 2. Prepare text chunks
# ------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, # Slightly smaller chunks work better for local models
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)

all_chunks_with_data = []

for page in pages:
    # Handle both dictionary and object structures from previous steps
    content_obj = page.get("content", {})
    if isinstance(content_obj, dict):
        content_md = content_obj.get("markdown", "")
        metadata_dict = content_obj.get("metadata", {})
    else:
        content_md = getattr(content_obj, 'markdown', "")
        metadata_dict = getattr(content_obj, 'metadata', {})

    if not content_md or not content_md.strip():
        continue

    chunks = text_splitter.split_text(content_md)

    for i, chunk in enumerate(chunks):
        all_chunks_with_data.append({
            "text": chunk,
            "metadata": {
                "url": page.get("url", ""),
                "title": metadata_dict.get("title", "Untitled"),
                "chunk_index": i
            }
        })

texts_to_embed = [item["text"] for item in all_chunks_with_data]
print(f"Total chunks created: {len(texts_to_embed)}")

# ------------------------------
# 3. Generate Local Embeddings
# ------------------------------
print("Loading local embedding model (all-MiniLM-L6-v2)...")
# This will download the model (~80MB) the first time you run it.
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Generating embeddings (this may take a minute)...")
embeddings = embeddings_model.embed_documents(texts_to_embed)
embeddings_np = np.array(embeddings).astype("float32")

# ------------------------------
# 4. Build and Save FAISS Index
# ------------------------------
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

faiss.write_index(index, FAISS_INDEX_PATH)
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks_with_data, f, indent=2, ensure_ascii=False)

print(f"Success! FAISS index saved with {index.ntotal} vectors.")