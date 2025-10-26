# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:26:33 2025

@author: chvuppal
"""

import cohere
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from typing import List, Tuple

from dotenv import load_dotenv
# Load environment variables
load_dotenv()
import os

# Configure API key
api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "API_KEY not found in .env file"
    )
# ---------- Setup ----------
co = cohere.ClientV2(api_key=api_key)

def image_to_base64_data_url(image_path):
    """Convert image to base64 data URL"""
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"

def image_to_base64_data_url_old(image_path: str) -> str:
    with Image.open(image_path) as img:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # assumes both are already L2-normalized
    return float(np.dot(a, b))

# ---------- Build image embedding store ----------
def embed_images(image_paths: List[str]) -> List[Tuple[str, np.ndarray]]:
    data_urls = [image_to_base64_data_url(p) for p in image_paths]
    res = co.embed(
        images=data_urls,
        model="embed-v4.0",
        embedding_types=["float"],
        input_type="image",
    )
    vecs = [np.array(v, dtype=np.float32) for v in res.embeddings.float]
    vecs = l2_normalize(np.stack(vecs, axis=0))
    return list(zip(image_paths, vecs))

# ---------- Natural-language search over images ----------
def search_images(
    query: str,
    image_index: List[Tuple[str, np.ndarray]],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    qres = co.embed(
        texts=[query],
        model="embed-v4.0",
        embedding_types=["float"],
        input_type="search_query",  # key for cross-modal retrieval
    )
    
    print("qres")
    embedding_vector = qres.embeddings.float[0]
    print("qres Embedding vector length:", len(embedding_vector))
    
    
    qvec = np.array(qres.embeddings.float[0], dtype=np.float32)
    qvec = l2_normalize(qvec)
    
    # --- Retrieve embedding and token count ---
    #embedding_vector = qres.embeddings.float[0] 

    scored = [] 
    for path, ivec in image_index:
        score = cosine_sim(qvec, ivec)
        scored.append((path, score)) 
        
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# 1) Embed a small corpus of images
# "IMG_5345.jpg",
# "0c241eba-1ac6-4a56-b260-d99723209d47_94.jpg",
image_paths = [
      "ADV_college-of-science_2.jpg" , "ADV_college-of-social-sciences_2.jpg",
]

image_index = embed_images(image_paths)  

print(image_index)


# 2) Run a natural-language query
#query = "person handling a package on a residential porch; delivery truck on the street"
#query = "Wells Fargo check"
#query = "glasses, necklace, hill with sun and fence"
#query = "Can we copy Strike ?"
query = "person with tape and cap"
results = search_images(query, image_index, top_k=3)

print(results)

# 3) Show results
print("\nTop matches:")
for path, score in results:
    print(f"{path}  |  cosine={score:.4f}")
        



