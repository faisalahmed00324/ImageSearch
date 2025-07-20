MONGO_URI       = "mongodb://localhost:27017"
DB_NAME         = "VisualSearch"
COLLECTION_NAME = "images"

INDEX_FILE      = ".\py-scripts\dino_index.faiss"
ID_FILE         = ".\py-scripts\mongo_ids.npy"

IMAGE_FOLDER    = "./images"          # put your jpg/png/etc. here

TOURCE_CACHE = ".\py-scripts/torch_cache"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"