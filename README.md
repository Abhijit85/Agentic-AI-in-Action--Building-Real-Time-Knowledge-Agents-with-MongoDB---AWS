# Agentic AI Demo

This repository now demonstrates a retrieval‑augmented workflow that streams
knowledge from Amazon S3, persists chunked embeddings into MongoDB Atlas, and
retrieves answers via Atlas Search hybrid (vector + keyword) queries.

## Components

- `demo.py` – orchestrates ingestion from S3, embedding, Atlas persistence, and
  the interactive question loop.
- `services/` – integration helpers for S3 streaming, text chunking, hashing
  embeddings, the MongoDB Atlas persistence layer, and Bedrock-backed LLM calls.
- `services/prompting.py` – shared prompt and context-normalisation helpers for
  both the CLI demo and AgentCore tooling.
- `agent/agentcore_config.py` – declarative AgentCore runtime, memory, identity
  and tool configuration for the retrieval-augmented pipeline.
- `agent/tool_definitions.json` – Model Context Protocol (MCP) tool definitions
  for the S3 ingest, MongoDB Atlas search, and Bedrock answer tools.

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
- `AWS_REGION` (required): used for both data access and Bedrock model calls.
- `AWS_ACCESS_KEY_ID` (required): access key for AWS API calls.
- `AWS_SECRET_ACCESS_KEY` (required): secret key paired with the access key.
- `AWS_SESSION_TOKEN` (optional): include when using temporary credentials.
- `BEDROCK_MODEL_ID` (default: `anthropic.claude-3-haiku-20240307-v1:0`).
- `LLM_MAX_TOKENS` (default: `512`), `LLM_TEMPERATURE` (default: `0.2`).
- `LLM_STREAMING` (default: `false`): when `true`, stream Bedrock tokens to the CLI.

The demo reads AWS credentials directly from the environment variables listed
above and will exit if they are missing.

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
   issues an Atlas Search compound `text` + `knnBeta` query, routes the
   retrieved context into an Amazon Bedrock model, and streams back a grounded
   answer alongside source citations and retrieval scores.

4. Type `exit` (or press `Ctrl+C`) to quit.

## Deploying with Bedrock AgentCore

This repository now includes everything required to host the retrieval pipeline
as a Bedrock AgentCore agent. The `agent/agentcore_config.py` module exposes a
single `build_agentcore_configuration` function that produces a complete
runtime/memory/identity/tool configuration compatible with AgentCore's SDK. The
same module also implements the three gateway tools referenced in the proposal:

- `s3_ingest_tool` uploads fresh content from Amazon S3, chunks and embeds the
  text, and upserts the vectors into MongoDB Atlas.
- `mongo_search_tool` performs a hybrid Atlas Search query using the shared
  hashing embedder to keep scoring consistent across the CLI and AgentCore.
- `bedrock_answer_tool` accepts retrieved context, builds a grounded prompt via
  `services.prompting.build_grounded_answer_prompt`, and calls the configured
  Amazon Bedrock model. Optional flags enable streaming and prompt inspection so
  the gateway can support interactive channels.

To register the tools with AgentCore Gateway, publish `agent/tool_definitions.json`.
The JSON mirrors the schemas used inside `build_tool_specs`, so the CLI demo and
the gateway stay in sync. When AgentCore is available at runtime the helper
`initialise_agent_runtime` dynamically registers the tool functions with the
runtime and wires in observability and security policies.

### Identity & Security

AgentCore Identity support is built in. Provide the following optional
environment variables to enable Cognito enforcement via AgentCore's managed
identity service:

- `AGENT_COGNITO_USER_POOL_ID`
- `AGENT_COGNITO_APP_CLIENT_ID`
- `AGENT_ALLOWED_GROUPS` (comma-separated)
- `AGENT_IDENTITY_PROVIDER` (override when using a non-Cognito identity stack)

When these are present `build_agentcore_configuration` injects an `identity`
block so AgentCore can validate tokens and group membership. The security policy
has also been updated to whitelist the new `bedrock_answer` gateway tool.

## Notes

- The embeddings are produced with a `HashingVectorizer`, enabling deterministic
  vectors without a separate model download. Swap in a higher quality embedding
  model if desired; ensure the Atlas Search index definition matches the new
  vector dimensionality.
- The Atlas index definition is created on startup if it does not already
  exist. Re-running the script will upsert chunks by a deterministic chunk ID
  (`<s3-key>:::<chunk-index>`).
- Answers are generated by the configured Bedrock model. Provide AWS
  credentials through `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and, when
  applicable, `AWS_SESSION_TOKEN`.
