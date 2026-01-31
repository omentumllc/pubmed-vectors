"""
Build a FAISS index from existing PubMed embeddings (HDF5 or Parquet).

This converts the brute-force search into a sub-linear ANN (Approximate
Nearest Neighbor) lookup. Build the index once, then query in milliseconds.

Index types:
  flat   - Exact search. No training. Best for <500K vectors.
  ivf    - IVF with flat quantizer. 500K-10M vectors. ~100x faster than flat.
  ivfsq  - IVF + scalar quantization (int8). 10M+ vectors. 4x less RAM than ivf.
  ivfpq  - IVF + product quantization. 10M+ vectors. ~16x less RAM than ivf.
  auto   - Automatically selects based on dataset size (default).

Usage:
    # From HDF5 (auto-selects index type):
    python build_faiss_index.py --source pubmed_embeddings.h5 --output ./faiss_index/

    # From Parquet with explicit index type:
    python build_faiss_index.py --source pubmed_embeddings.parquet --output ./faiss_index/ --index-type ivfsq

    # From Parquet on S3 (requires s3fs):
    python build_faiss_index.py --source s3://bucket/pubmed_embeddings.parquet --output ./faiss_index/

Recommended index types by dataset size:
    3M vectors  -> ivf    (~3GB index, ~2ms queries, >95% recall)
    36M vectors -> ivfsq  (~7GB index, ~5ms queries, >90% recall)
    36M vectors -> ivfpq  (~2GB index, ~5ms queries, >85% recall)
"""

import argparse
import faiss
import numpy as np
import os
import json
from tqdm import tqdm


def normalize_vectors(vectors):
    """L2-normalize vectors so inner product = cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def select_nlist(num_vectors):
    """Select number of IVF clusters based on dataset size.

    Uses the FAISS-recommended heuristic of 4*sqrt(n), capped for
    practical memory and training time constraints.
    """
    nlist = int(4 * np.sqrt(num_vectors))
    # Cap based on practical limits
    if num_vectors < 1_000_000:
        return min(nlist, 1024)
    elif num_vectors < 10_000_000:
        return min(nlist, 4096)
    else:
        return min(nlist, 8192)


def create_index(dim, num_vectors, index_type='auto'):
    """Create a FAISS index with appropriate parameters.

    Returns:
        index: FAISS index object
        needs_training: Whether the index needs to be trained before adding vectors
        index_type: The actual index type used (resolved from 'auto')
    """
    if index_type == 'auto':
        if num_vectors < 500_000:
            index_type = 'flat'
        elif num_vectors < 10_000_000:
            index_type = 'ivf'
        else:
            index_type = 'ivfsq'

    if index_type == 'flat':
        # Exact search using inner product (cosine sim on normalized vectors)
        index = faiss.IndexFlatIP(dim)
        needs_training = False
        print(f"Index type: flat (exact search)")

    elif index_type == 'ivf':
        nlist = select_nlist(num_vectors)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
        )
        needs_training = True
        print(f"Index type: ivf (nlist={nlist})")

    elif index_type == 'ivfpq':
        nlist = select_nlist(num_vectors)
        # m=48 sub-quantizers, each encoding 768/48=16 dimensions
        # nbits=8 gives 256 centroids per sub-quantizer
        # Compressed vector size: 48 bytes (vs 3072 bytes for float32)
        m = 48
        nbits = 8
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
        )
        needs_training = True
        print(f"Index type: ivfpq (nlist={nlist}, m={m}, nbits={nbits})")

    elif index_type == 'ivfsq':
        nlist = select_nlist(num_vectors)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT
        )
        needs_training = True
        print(f"Index type: ivfsq (nlist={nlist}, scalar int8 quantization)")

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    return index, needs_training, index_type


def get_training_vectors(source_file, num_vectors, train_size, batch_size):
    """Sample training vectors from the source file."""
    if source_file.endswith('.h5') or source_file.endswith('.hdf5'):
        return _sample_h5(source_file, num_vectors, train_size)
    else:
        return _sample_parquet(source_file, train_size, batch_size)


def _sample_h5(h5_file, num_vectors, train_size):
    """Sample evenly-spaced vectors from HDF5 for training."""
    import h5py
    indices = np.linspace(0, num_vectors - 1, train_size, dtype=int)
    with h5py.File(h5_file, 'r') as f:
        # Read in sorted order for sequential HDF5 access
        sorted_idx = np.sort(indices)
        vectors = f['doc_vectors'][sorted_idx].astype('float32')
    return normalize_vectors(vectors)


def _sample_parquet(parquet_file, train_size, batch_size):
    """Read vectors from the beginning of the Parquet file for training."""
    collected = []
    total = 0
    for _, vecs in iter_parquet(parquet_file, batch_size):
        collected.append(vecs)
        total += len(vecs)
        if total >= train_size:
            break
    vectors = np.vstack(collected)[:train_size]
    return normalize_vectors(vectors)


def iter_parquet(parquet_file, batch_size=100_000):
    """Yield (ids, vectors) batches from a Parquet file."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    if parquet_file.startswith('s3://'):
        import s3fs
        fs = s3fs.S3FileSystem()
        reader = pq.ParquetFile(fs.open(parquet_file))
    else:
        reader = pq.ParquetFile(parquet_file)

    for batch in reader.iter_batches(batch_size=batch_size):
        df = pa.Table.from_batches([batch]).to_pandas()
        ids = df['doc_id'].values.astype('int64')
        vectors = np.vstack(df['doc_vector'].values).astype('float32')
        yield ids, vectors


def get_source_info(source_file):
    """Get num_vectors and dimension from the source file."""
    if source_file.endswith('.h5') or source_file.endswith('.hdf5'):
        import h5py
        with h5py.File(source_file, 'r') as f:
            num_vectors = f['doc_vectors'].shape[0]
            dim = f['doc_vectors'].shape[1]
            doc_ids = f['doc_ids'][:]
        return num_vectors, dim, doc_ids
    else:
        import pyarrow.parquet as pq
        import pyarrow as pa

        if source_file.startswith('s3://'):
            import s3fs
            fs = s3fs.S3FileSystem()
            pf = pq.ParquetFile(fs.open(source_file))
        else:
            pf = pq.ParquetFile(source_file)

        num_vectors = pf.metadata.num_rows
        first_batch = next(pf.iter_batches(batch_size=1))
        df = pa.Table.from_batches([first_batch]).to_pandas()
        dim = len(df['doc_vector'].iloc[0])
        return num_vectors, dim, None  # IDs read during iteration for Parquet


def build_index(source_file, output_dir, index_type='auto', batch_size=100_000):
    """Build a FAISS index from HDF5 or Parquet embeddings."""

    print(f"Source: {source_file}")
    num_vectors, dim, preloaded_ids = get_source_info(source_file)
    print(f"Dataset: {num_vectors:,} vectors, {dim} dimensions")

    index, needs_training, actual_type = create_index(dim, num_vectors, index_type)

    # --- Train ---
    if needs_training:
        nlist = index.nlist if hasattr(index, 'nlist') else 256
        train_size = min(num_vectors, max(256 * nlist, 100_000))
        print(f"Training index with {train_size:,} sampled vectors...")
        train_vectors = get_training_vectors(source_file, num_vectors, train_size, batch_size)
        index.train(train_vectors)
        del train_vectors
        print("Training complete.")

    # --- Add vectors ---
    print(f"Adding {num_vectors:,} vectors to index...")

    if source_file.endswith('.h5') or source_file.endswith('.hdf5'):
        import h5py
        with h5py.File(source_file, 'r') as f:
            for start in tqdm(range(0, num_vectors, batch_size)):
                end = min(start + batch_size, num_vectors)
                vectors = f['doc_vectors'][start:end].astype('float32')
                vectors = normalize_vectors(vectors)
                index.add(vectors)
        doc_ids = preloaded_ids
    else:
        all_ids = []
        total_batches = (num_vectors + batch_size - 1) // batch_size
        for ids_batch, vecs_batch in tqdm(
            iter_parquet(source_file, batch_size),
            total=total_batches
        ):
            vecs_batch = normalize_vectors(vecs_batch)
            index.add(vecs_batch)
            all_ids.append(ids_batch)
        doc_ids = np.concatenate(all_ids)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, 'pubmed.faiss')
    id_map_path = os.path.join(output_dir, 'pubmed_ids.npy')

    print(f"Saving index ({index.ntotal:,} vectors)...")
    faiss.write_index(index, index_path)
    np.save(id_map_path, doc_ids)

    metadata = {
        'index_type': actual_type,
        'num_vectors': int(num_vectors),
        'dimension': int(dim),
        'source': source_file,
    }
    with open(os.path.join(output_dir, 'index_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    index_size_gb = os.path.getsize(index_path) / 1e9
    id_map_size_mb = os.path.getsize(id_map_path) / 1e6

    print(f"\nIndex saved to {output_dir}/")
    print(f"  pubmed.faiss:   {index_size_gb:.2f} GB")
    print(f"  pubmed_ids.npy: {id_map_size_mb:.1f} MB")
    print(f"  Total vectors:  {index.ntotal:,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build FAISS index from PubMed embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_faiss_index.py --source pubmed_embeddings.h5 --output ./faiss_index/
  python build_faiss_index.py --source pubmed_embeddings.parquet --output ./faiss_index/ --index-type ivfpq
  python build_faiss_index.py --source s3://bucket/embeddings.parquet --output ./faiss_index/
        """
    )
    parser.add_argument(
        '--source', required=True,
        help='Path to HDF5 (.h5/.hdf5) or Parquet (.parquet) file. S3 paths supported for Parquet.'
    )
    parser.add_argument(
        '--output', default='./faiss_index/',
        help='Output directory for the FAISS index files (default: ./faiss_index/)'
    )
    parser.add_argument(
        '--index-type', default='auto',
        choices=['auto', 'flat', 'ivf', 'ivfpq', 'ivfsq'],
        help='Index type. "auto" selects based on dataset size (default: auto)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=100_000,
        help='Batch size for reading and adding vectors (default: 100000)'
    )

    args = parser.parse_args()

    ext = os.path.splitext(args.source)[-1].lower()
    if ext not in ('.h5', '.hdf5', '.parquet') and not args.source.startswith('s3://'):
        parser.error(f"Unsupported file format: {args.source} (use .h5, .hdf5, or .parquet)")

    build_index(args.source, args.output, args.index_type, args.batch_size)
