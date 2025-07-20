# ImageSearch

A simple visual search engine for images using DINOv2 embeddings, FAISS for similarity search, and MongoDB for metadata storage.

## Features

- **Add images**: Embed and index new images with [`add_image.py`](py-scripts/add_image.py).
- **Build index**: Bulk index all images in a folder with [`build_index.py`](py-scripts/build_index.py).
- **Search**: Find visually similar images using [`search.py`](py-scripts/search.py).

## Setup

1. **Install dependencies**:
   - Python 3.8+
   - `torch`, `faiss`, `numpy`, `pillow`, `pymongo`, `timm`
2. **Start MongoDB**:
   - Ensure a local MongoDB instance is running (`mongodb://localhost:27017`).

3. **Configure paths**:
   - Edit [`cfg.py`](py-scripts/cfg.py) to set image folder and cache locations.

## Usage

### 1. Build the index

Put your images in the folder specified by `IMAGE_FOLDER` in [`cfg.py`](py-scripts/cfg.py) (default: `./images`).


python [build_index.py]

Add a new image
python [add_image.py] <image_path> [image_id]

Search for similar images
python [search.py] <query_image_path> --k 10