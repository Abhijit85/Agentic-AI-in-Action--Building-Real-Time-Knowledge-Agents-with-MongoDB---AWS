# Agentic AI Demo

This repository now demonstrates a retrieval‑augmented workflow that streams
knowledge from Amazon S3, persists chunked embeddings into MongoDB Atlas, and
retrieves answers via Atlas Search hybrid (vector + keyword) queries.

## Components

- `demo.py` – orchestrates ingestion from S3, embedding, Atlas persistence, and
  the interactive question loop.
- `services/` – integration helpers for S3 streaming, text chunking, hashing
  embeddings, and the MongoDB Atlas persistence layer.

## Requirements

- Python 3.8 or later
- `boto3`
- `pymongo`
- `scikit-learn`
- `numpy`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Configuration

Provide the following environment variables before running the demo:

- `S3_BUCKET_NAME` (required): name of the source bucket.
- `S3_PREFIX` (optional): restrict ingestion to keys under this prefix.
- `MONGODB_ATLAS_URI` (required): connection string with write access.
- `ATLAS_DB_NAME` (default: `demo`)
- `ATLAS_COLLECTION_NAME` (default: `documents`)
- `ATLAS_SEARCH_INDEX_NAME` (default: `demo_rag_index`)
- `CHUNK_WORDS` (default: `400`): chunk size in words.
- `CHUNK_OVERLAP` (default: `40`): overlap between consecutive chunks.
- `TOP_K` (default: `3`): number of results returned per query.

AWS credentials are resolved via the usual boto3 lookup chain (environment
variables, AWS config files, or assumed roles).

## Running the Demo

1. Export the required environment variables.
2. Ensure the target MongoDB Atlas database allows the IP you are running from
   and that the Atlas Search feature is available for the cluster tier.
3. Execute the script:

   ```bash
   python demo.py
   ```

   The script will ingest objects from S3, chunk and embed them, upsert the
   chunks into MongoDB Atlas, and then start an interactive prompt. Each query
   issues an Atlas Search compound `text` + `knnBeta` query and displays
   separate keyword and vector scores.

4. Type `exit` (or press `Ctrl+C`) to quit.

## Notes

- The embeddings are produced with a `HashingVectorizer`, enabling deterministic
  vectors without a separate model download. Swap in a higher quality embedding
  model if desired; ensure the Atlas Search index definition matches the new
  vector dimensionality.
- The Atlas index definition is created on startup if it does not already
  exist. Re-running the script will upsert chunks by a deterministic chunk ID
  (`<s3-key>:::<chunk-index>`).
- When adapting this demo, consider wiring in a downstream LLM or answering
  module that consumes the retrieved context for richer responses.
