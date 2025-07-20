import os
import argparse
import numpy as np
import faiss
import torch
from PIL import Image
from pymongo import MongoClient
from bson import ObjectId
import cfg
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message="xFormers is not available*")
if  cfg.TOURCE_CACHE:
    os.environ['TORCH_HOME'] = cfg.TOURCE_CACHE
# Load DINOv2 model and transform (same as before)
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(cfg.DEVICE)
model.eval()
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
transform = create_transform(**resolve_data_config({}, model=model))

def embed(img_path):
    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        vec = model(img)
    return vec.cpu().numpy().flatten().astype("float32")

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("--k", type=int, default=10)
args = ap.parse_args()

index = faiss.read_index(cfg.INDEX_FILE)
mongo_ids = list(np.load(cfg.ID_FILE, allow_pickle=True))
client = MongoClient(cfg.MONGO_URI)
col = client[cfg.DB_NAME][cfg.COLLECTION_NAME]

q_vec = embed(args.query)
faiss.normalize_L2(q_vec.reshape(1, -1))

D, I = index.search(q_vec.reshape(1, -1), args.k)



max_sim = 0
for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
    if idx == -1:
        continue
    if score < 0.1:
        continue
    if score > max_sim:
        max_sim = score
    doc_id = mongo_ids[idx]
    doc = col.find_one({"_id": ObjectId(doc_id)})
    if doc is None:        
        continue
    print(f"{rank};cos_sim={score:.4f};{doc['imageId']};{doc['image_path']}")




