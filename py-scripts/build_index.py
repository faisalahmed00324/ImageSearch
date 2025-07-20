
print("start")
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available*")
import os, numpy as np, faiss, torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
from pymongo import MongoClient
import cfg

# ---------- 1.  load DINO-v2 ViT-S/14 ----------
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(cfg.DEVICE)
model.eval()
print("✓ DINO-v2 loaded")
# build the same transform that DINO-v2 was trained with
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
transform = create_transform(**resolve_data_config({}, model=model))

def embed(path: str) -> np.ndarray:
    img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        vec = model(img)            # (1,384)
    return vec.cpu().numpy().flatten().astype("float32")

# ---------- 2.  MongoDB ----------
client  = MongoClient(cfg.MONGO_URI, serverSelectionTimeoutMS=2000)
col     = client[cfg.DB_NAME][cfg.COLLECTION_NAME]
col.delete_many({})                # optional: wipe previous docs

ids, vectors = [], []
for f in sorted(os.listdir(cfg.IMAGE_FOLDER)):
    if f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff")):
        p   = os.path.join(cfg.IMAGE_FOLDER, f)
        vec = embed(p)
        _id = col.insert_one({"image_path": p,
                              "vector": vec.tolist(),
                              "metadata": {}}).inserted_id
        ids.append(str(_id))
        vectors.append(vec)
print(f"Indexed {len(ids)} photos.")

# ---------- 3.  FAISS ----------
vectors = np.vstack(vectors)
faiss.normalize_L2(vectors)        # cosine similarity
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, cfg.INDEX_FILE)
np.save(cfg.ID_FILE, np.array(ids))
print("Saved FAISS index →", cfg.INDEX_FILE)


