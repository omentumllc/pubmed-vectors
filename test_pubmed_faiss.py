"""
Interactive query test using FAISS-based retriever.

Usage:
    python test_pubmed_faiss.py
    python test_pubmed_faiss.py --index-dir ./faiss_index/ --db pubmed_abstracts_2024.db
    python test_pubmed_faiss.py --no-onnx  # disable ONNX, use PyTorch backend
"""

import argparse
import logging
import time

from abstract_retriever_faiss import AbstractRetrieverFaiss


def main():
    parser = argparse.ArgumentParser(description='Query PubMed abstracts with FAISS')
    parser.add_argument('--index-dir', default='./faiss_index/',
                        help='Directory containing pubmed.faiss and pubmed_ids.npy')
    parser.add_argument('--db', default='pubmed_abstracts_2024.db',
                        help='Path to SQLite database')
    parser.add_argument('--model', default='nomic-ai/nomic-embed-text-v1.5',
                        help='Embedding model name')
    parser.add_argument('--no-onnx', action='store_true',
                        help='Disable ONNX Runtime (use PyTorch backend)')
    parser.add_argument('--nprobe', type=int, default=32,
                        help='Number of IVF clusters to probe (higher=better recall, slower)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of results to return')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    index_path = f"{args.index_dir}/pubmed.faiss"
    id_map_path = f"{args.index_dir}/pubmed_ids.npy"

    print("Loading retriever...")
    t0 = time.perf_counter()
    retriever = AbstractRetrieverFaiss(
        index_path, id_map_path, args.db,
        model_name=args.model,
        use_onnx=not args.no_onnx,
        nprobe=args.nprobe
    )
    load_time = time.perf_counter() - t0
    print(f"Retriever loaded in {load_time:.1f}s")
    print(f"  Embedding backend: {retriever._backend}")
    print(f"  Index vectors: {retriever.index.ntotal:,}")
    print(f"  nprobe: {args.nprobe}")
    print()

    while True:
        query = input("Enter your query: ").strip()

        if query == "":
            query = "What is the role of GLP-1 and GLP-1 agonists in losing excess weight?"

        if query.lower() == "exit":
            break

        print(f'Looking up PubMed abstracts using "{query}"...')

        t0 = time.perf_counter()
        pmids, similarities, documents = retriever.search(query, top_k=args.top_k)
        elapsed = time.perf_counter() - t0

        if len(documents) > 0:
            # Results come back already sorted by FAISS (descending similarity)
            for i, (similarity, doc) in enumerate(zip(similarities, documents)):
                print(f"Rank {i + 1}, Similarity: {similarity:.4f}")
                print(f"PMID: {doc['pmid']}")
                print(f"Title: {doc['title']}")
                print(f"Authors: {doc['authors']}")
                print(f"Abstract: {doc['abstract']}")
                print(f"Publication Year: {doc['publication_year']}")
                print("-----")

        print(f"\nTotal search time: {elapsed*1000:.1f}ms")
        print()


if __name__ == '__main__':
    main()
