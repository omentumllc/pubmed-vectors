FROM python:3.11-slim

# System deps needed by faiss-cpu and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer - only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .

# Pre-download the embedding model so first query doesn't stall.
# This adds ~500MB to the image but avoids a download on every container start.
# The model is cached in /root/.cache/huggingface/
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True); \
print(f'Model loaded: {m.get_sentence_embedding_dimension()} dims')"

# Data directory - mount your HDF5/Parquet, SQLite DB, and FAISS index here
VOLUME /data

# HuggingFace cache - mount to persist model downloads and ONNX exports across restarts
VOLUME /root/.cache/huggingface

ENTRYPOINT ["python"]
CMD ["test_pubmed_faiss.py", "--index-dir", "/data/faiss_index", "--db", "/data/pubmed_abstracts_2024.db"]
