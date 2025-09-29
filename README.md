# Agentic AI Demo

This repository contains a simple demonstration of an agentic AI knowledge assistant
built without external APIs.  It illustrates how retrieval‑augmented generation
(RAG) pipelines can be assembled using off‑the‑shelf Python packages.

## Contents

- `demo.py` – the main script that loads a small knowledge base of text files,
  builds a TF‑IDF index, accepts natural language questions, retrieves the most
  relevant documents and produces a basic summary.
- `data/` – a directory containing sample documents about topics such as
  Matryoshka embeddings, Voyage AI models, Amazon Bedrock AgentCore and
  retrieval‑augmented generation.  The script reads all `.txt` files in this
  directory as its knowledge base.

## Requirements

This demo relies on the following Python packages, which are included in the
exercise environment:

- Python 3.7 or later
- `scikit‑learn`
- `numpy`

No network access or large language model is required.  The script uses
TF‑IDF vectors for semantic search and a simple word‑frequency algorithm for
summarization.

## Running the Demo

1. Ensure that `demo.py` and the `data/` directory are in the same folder.
2. Open a terminal and run:

   ```bash
   python demo.py
   ```

3. When prompted, type a question related to the knowledge base, e.g.:

   ```
   What is Matryoshka representation learning?
   What is AgentCore and what does it provide?
   How do embedding models and rerankers reduce hallucinations?
   ```

   The script will display the top documents it retrieved and then
   output a brief answer assembled from the relevant documents.

4. Type `exit` or press `Ctrl+C` to quit.

## Extending the Demo

This demo is intentionally simple.  To adapt it for a real workshop:

- Replace the sample documents in `data/` with your own knowledge base or
  ingest data from other sources (e.g. PDFs, web pages).  You can split large
  documents into paragraphs or chunks and store each as a separate text file.
- Use a more sophisticated vectorizer (such as sentence‑transformers) and
  summarization model if you have access to larger models and network resources.
- Integrate with a database like MongoDB to persist the knowledge base and
  connect the retrieval pipeline to a vector search index.
- Incorporate a generative language model (e.g. via Amazon Bedrock) to
  formulate answers using the retrieved context.

Feel free to modify and extend this script to suit your workshop needs.
