#!/usr/bin/env python3
# add_image.py
# -----------------------------------------------------------
# Add ONE new image to the visual-search dataset (MongoDB + FAISS)
# Usage:  python add_image.py <image_path> [image_id]
#          image_id is optional; if provided it will be saved
#          in the document as "image_id"
# -----------------------------------------------------------

import os, sys, warnings, numpy as np, faiss, torch
from PIL import Image
from pymongo import MongoClient
from bson import ObjectId
import cfg

# --- housekeeping -----------------------------------------------------------

if len(sys.argv) not in (2, 3):
    sys.exit("Usage:  python add_image.py <image_path> [image_id]")

img_path = sys.argv[1]
image_id = sys.argv[2] if len(sys.argv) == 3 else None

if not os.path.isfile(img_path):
    sys.exit(f"File not found: {img_path}")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # avoid OMP clash
warnings.filterwarnings("ignore", message="xFormers is not available*")

# --- load DINO-v2 encoder ----------------------------------------------------

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(cfg.DEVICE)
model.eval()

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
transform = create_transform(**resolve_data_config({}, model=model))

def embed(path: str) -> np.ndarray:
    """Return L2‑normalised 384‑D DINOv2 embedding as float32."""
    img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        vec = model(img)
    vec = vec.cpu().numpy().astype("float32").flatten()
    # L2‑normalise so FAISS Inner‑Product acts like cosine sim
    vec /= np.linalg.norm(vec) + 1e-8
    return vec

# --- connect storage ---------------------------------------------------------

client = MongoClient(cfg.MONGO_URI)
col    = client[cfg.DB_NAME][cfg.COLLECTION_NAME]

# load / create FAISS index
vec = embed(img_path)
_dim = vec.shape[0]

if os.path.exists(cfg.INDEX_FILE):
    index = faiss.read_index(cfg.INDEX_FILE)
else:
    index = faiss.IndexFlatIP(_dim)        # cosine via normalised IP

# load / create id list
if os.path.exists(cfg.ID_FILE):
    mongo_ids = list(np.load(cfg.ID_FILE, allow_pickle=True))
else:
    mongo_ids = []

# --- write to MongoDB --------------------------------------------------------

doc = {
    "imageId": image_id,
    "image_path": img_path,
    "vector"    : vec.tolist(),
    "metadata"  : {},           # extend as you wish
}



inserted_id = col.insert_one(doc).inserted_id
mongo_ids.append(str(inserted_id))

# --- add to FAISS ------------------------------------------------------------

index.add(vec.reshape(1, -1))

# --- persist -----------------------------------------------------------------

faiss.write_index(index, cfg.INDEX_FILE)
np.save(cfg.ID_FILE, np.array(mongo_ids))

print(f"✓ Added {img_path}")
print(f"  _id      : {inserted_id}")
print(f"  image_id : {image_id}")
print(f"  new size : {index.ntotal} vectors")
