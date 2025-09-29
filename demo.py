#!/usr/bin/env python3
"""
Demo script for the Agentic AI workshop.

This script demonstrates a simplified retrieval‑augmented generation pipeline
without external dependencies like large language models.  It uses TF‑IDF to
create embeddings for a small set of knowledge documents and performs semantic
search to retrieve the most relevant documents for a given query.  It then
produces a basic summary of the retrieved documents using a simple word‑frequency
scoring algorithm.

Usage:
    python demo.py

The script will load text files from the `data` directory relative to its
location and then enter an interactive loop where you can type questions.
Press Ctrl+C or type 'exit' to quit.

This demo does not require network access and can run on modest hardware.
"""
import os
import re
import math
import sys
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents(data_dir: Path) -> Tuple[List[str], List[str]]:
    """Load text documents from a directory.

    Args:
        data_dir: Path to the directory containing text files.

    Returns:
        A tuple (docs, names) where docs is a list of document contents and names
        is a list of corresponding file names.
    """
    docs = []
    names = []
    for file in sorted(data_dir.glob("*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            docs.append(f.read().strip())
        names.append(file.name)
    return docs, names


def build_vectorizer(documents: List[str]) -> Tuple[TfidfVectorizer, any]:
    """Fit a TF‑IDF vectorizer on the provided documents.

    Args:
        documents: A list of strings to vectorize.

    Returns:
        A tuple (vectorizer, matrix) where vectorizer is the fitted
        TfidfVectorizer instance and matrix is the TF‑IDF matrix of the
        documents.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def search(query: str, vectorizer: TfidfVectorizer, matrix, doc_names: List[str], top_k: int = 3) -> List[Tuple[str, float, str]]:
    """Perform semantic search over the knowledge base.

    Args:
        query: The user question.
        vectorizer: The fitted vectorizer.
        matrix: The TF‑IDF matrix of the knowledge base.
        doc_names: Names corresponding to documents.
        top_k: Number of top documents to return.

    Returns:
        A list of tuples (document_name, score, content) of the top_k most
        similar documents.
    """
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    # Get indices of top documents
    top_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append((doc_names[idx], scores[idx], vectorizer.inverse_transform(matrix[idx])))
    return [(doc_names[idx], scores[idx], matrix[idx]) for idx in top_indices]


def retrieve_contents(query: str, vectorizer: TfidfVectorizer, matrix, documents: List[str], doc_names: List[str], top_k: int = 3) -> List[Tuple[str, float, str]]:
    """Return the top_k documents and their contents for a query."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append((doc_names[idx], float(scores[idx]), documents[idx]))
    return results


def simple_summarize(texts: List[str], num_sentences: int = 3) -> str:
    """Summarize a list of documents using a simple word frequency heuristic.

    This function splits the input documents into sentences, scores each
    sentence based on the frequency of its words (excluding common stop words),
    and returns the top sentences as a summary.

    Args:
        texts: A list of strings to summarize.
        num_sentences: The maximum number of sentences to include in the summary.

    Returns:
        A summarised string containing the selected sentences in their original
        order.
    """
    # Define a simple list of stopwords
    stop_words = set([
        'the', 'is', 'and', 'to', 'of', 'a', 'in', 'for', 'on', 'with', 'that', 'this',
        'as', 'an', 'by', 'it', 'be', 'are', 'from', 'or', 'at', 'into', 'their',
        'which', 'these', 'such'
    ])

    # Split texts into sentences using a simple regex
    sentences = []
    for text in texts:
        # Ensure consistent spacing
        text = text.replace('\n', ' ').strip()
        parts = re.split(r'(?<=[\.\?!])\s+', text)
        for sentence in parts:
            sentence = sentence.strip()
            if len(sentence.split()) > 4:  # ignore very short sentences
                sentences.append(sentence)

    if not sentences:
        return "".join(texts)

    # Compute word frequencies
    word_freq = {}
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

    # Score sentences
    sentence_scores = []
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(word_freq.get(word, 0) for word in words)
        sentence_scores.append((sentence, score))

    # Select top sentences by score
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    # Preserve original order
    top_sentences_text = [ts[0] for ts in sorted(top_sentences, key=lambda x: sentences.index(x[0]))]
    return " ".join(top_sentences_text)


def main():
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found.")
        sys.exit(1)

    # Load documents
    documents, doc_names = load_documents(data_dir)
    if not documents:
        print(f"No text files found in {data_dir}.")
        sys.exit(1)

    print(f"Loaded {len(documents)} documents. Building vectorizer...")
    vectorizer, matrix = build_vectorizer(documents)
    print("Vectorizer built. You can now ask questions.")

    # Interactive loop
    try:
        while True:
            query = input("\nEnter your question (or type 'exit' to quit): ").strip()
            if not query or query.lower() in {"exit", "quit"}:
                break
            # Retrieve top documents
            results = retrieve_contents(query, vectorizer, matrix, documents, doc_names, top_k=3)
            print("\nTop documents:")
            for name, score, content in results:
                print(f"- {name} (score={score:.3f})")
            # Summarize
            summary = simple_summarize([content for _, _, content in results], num_sentences=3)
            print("\nAnswer:")
            print(summary)
    except KeyboardInterrupt:
        pass
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
