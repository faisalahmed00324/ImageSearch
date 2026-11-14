# ImageSearch

A personal proof of concept exploring **content-based image retrieval** using deep learning and vector similarity search.

ImageSearch uses **DINOv2** to generate image embeddings, **FAISS** to find visually similar images, and **MongoDB** to store image metadata and embedding information.

The goal is to explore how computer vision and vector search can be combined to build an image-based search engine.

## Architecture

```text
                  Input Images
                       |
                       v
                DINOv2 (ViT-S/14)
                       |
                       v
                Image Embeddings
                  (384 dimensions)
                       |
              +--------+--------+
              |                 |
              v                 v
          MongoDB             FAISS
       Metadata & Vectors   Vector Index
              |                 |
              +--------+--------+
                       |
                       v
                 Similarity Search
                       |
                       v
                Ranked Results
```
## Technical Decisions

### 1. Why DINOv2?

DINOv2 was selected as the image feature extraction model because it produces meaningful visual representations that capture semantic and visual similarities between images.

The ViT-S/14 model generates a 384-dimensional embedding for each image. These embeddings allow images to be compared based on their visual content rather than filenames or manually assigned tags.

### 2. Why FAISS IndexFlatIP?

FAISS was chosen for efficient similarity searching across image embeddings.

The project uses `IndexFlatIP`, which performs exact inner-product similarity comparisons. This provides accurate similarity rankings without introducing the complexity of approximate nearest-neighbor indexing.

### 3. Why Normalize Embeddings?

All embeddings are L2-normalized before similarity comparison.

With normalized vectors, inner-product similarity is equivalent to cosine similarity:

\[
\operatorname{similarity}(A,B)=A\cdot B
\]

This makes it possible to rank images based on the direction of their embedding vectors rather than their magnitude.

### 4. Why MongoDB?

MongoDB stores image metadata and their corresponding embeddings.

FAISS handles vector similarity searches, while MongoDB maintains the associated image information. This separation allows the project to retrieve image metadata after identifying similar vectors.

### 5. Current Limitations

- This is a proof-of-concept implementation.
- FAISS index and MongoDB records require synchronization.
- The project currently uses exact similarity search rather than approximate indexing.
- There is no web interface or REST API.
- Index persistence and recovery require additional consideration for larger datasets.

  
### Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Vision model | DINOv2 (ViT-S/14) |
| Deep learning | PyTorch |
| Vector search | FAISS |
| Database | MongoDB |
| Image processing | Pillow |
| Model utilities | timm |

## Features

- **Visual Search:** Find images that are visually similar to a query image.
- **Image Embeddings:** Generate 384-dimensional feature vectors using DINOv2.
- **Bulk Indexing:** Process images from a directory and build a searchable index.
- **Incremental Indexing:** Add individual images to an existing FAISS index.
- **Metadata Storage:** Store image paths, identifiers, metadata, and embeddings in MongoDB.
- **Cosine Similarity:** Use normalized embeddings and FAISS inner-product search to retrieve similar images.

## How It Works

1. DINOv2 processes each image and generates a feature embedding.
2. Embeddings are normalized to support cosine similarity.
3. FAISS stores the vectors in an `IndexFlatIP` index.
4. MongoDB stores the corresponding image records and metadata.
5. When a query image is provided, its embedding is generated and compared against the indexed vectors.
6. The most similar images are returned in ranked order.

## Project Structure

```text
ImageSearch/
├── images/
│   └──                  # Input image collection
├── py-scripts/
│   ├── cfg.py           # Configuration
│   ├── build_index.py   # Bulk image indexing
│   ├── add_image.py     # Add an individual image
│   ├── search.py        # Visual similarity search
│   ├── dino_index.faiss # Generated FAISS index
│   └── mongo_ids.npy    # Generated MongoDB ID mapping
└── README.md
```

Generated index files are created when the indexing scripts run.

## Prerequisites

- Python environment compatible with PyTorch and the selected dependencies
- MongoDB running locally or at a configured MongoDB URI
- PyTorch-compatible CPU or CUDA environment
- Internet access for the initial DINOv2 model download, unless the model is already cached

## Installation

Install the Python dependencies:

```bash
pip install torch torchvision timm faiss-cpu numpy pillow pymongo
```

Start MongoDB and configure the connection and storage paths in:

```text
py-scripts/cfg.py
```

Important configuration values:

- `MONGO_URI`: MongoDB connection string
- `DB_NAME`: Database name
- `COLLECTION_NAME`: Collection name
- `IMAGE_FOLDER`: Directory containing source images
- `INDEX_FILE`: FAISS index file location
- `ID_FILE`: MongoDB ID mapping file location

Ensure the configured file paths are valid for your operating system.

## Usage

Run the following commands from the repository root.

### 1. Build the image index

Place your images in the directory configured by `IMAGE_FOLDER`.

```bash
python py-scripts/build_index.py
```

This process:

- Loads the DINOv2 model.
- Generates embeddings for supported images.
- Stores image information and vectors in MongoDB.
- Builds and saves the FAISS similarity index.
- Saves the MongoDB ID mapping.

Bulk-indexed images use their filename without the extension as `imageId` (for example, `example.jpg` becomes `example`), which is included in search results.

**Important:** The current bulk indexing script clears the configured MongoDB collection before rebuilding the index. Use it only with a dedicated POC dataset.

### 2. Add an individual image

```bash
python py-scripts/add_image.py path/to/image.jpg
```

Optionally provide an image identifier:

```bash
python py-scripts/add_image.py path/to/image.jpg image-001
```

The image embedding is generated and added to MongoDB and FAISS.

### 3. Search for visually similar images

```bash
python py-scripts/search.py path/to/query-image.jpg --k 10
```

The search returns up to 10 similar images, including their similarity scores and stored image information.

Example output format:

```text
1;cos_sim=0.9234;image-001;./images/example.jpg
2;cos_sim=0.8871;image-002;./images/example2.jpg
```

*Example values are illustrative, not measured results.*

## Current Scope and Limitations

This repository is a technical proof of concept, not a production-ready search service.

Current implementation characteristics:

- Uses FAISS `IndexFlatIP`, which performs exact similarity search.
- Uses local FAISS index and MongoDB storage.
- Uses DINOv2 ViT-S/14 as the image embedding model.
- Does not currently provide a web-based user interface or REST API.
- Requires the FAISS index and MongoDB ID mapping to remain synchronized.

## Future Improvements

Potential areas for further exploration:

- Expose search functionality through a REST API.
- Build a React-based image search interface.
- Add image upload and management capabilities.
- Explore approximate nearest-neighbor indexing for larger datasets.
- Evaluate alternative vision embedding models.
- Improve metadata filtering and search result presentation.

## Disclaimer

This project was developed as an independent technical proof of concept to explore image embeddings and vector similarity search. It is intended for experimentation and learning.
