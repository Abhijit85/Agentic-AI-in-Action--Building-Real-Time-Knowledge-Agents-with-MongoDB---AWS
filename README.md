# Agentic AI Demo

This repository now demonstrates a media-focused retrieval‑augmented workflow
that combines Amazon S3 ingestion, the [YouTubeRAG](https://github.com/FernandoBDAF/YoutubeRAG)
pipeline, MongoDB Atlas vector search, and Amazon Bedrock generation. The agent
can ingest high-value YouTube playlists, persist diarised transcript segments
with 1024‑dimensional embeddings, and answer grounded questions that reference
video titles, timestamps, speakers, and channels.

## Components

- `demo.py` – orchestrates ingestion from S3, embedding, Atlas persistence, and
  the interactive question loop.
- `services/` – integration helpers for S3 streaming, text chunking, Voyage or
  hashing embeddings, the MongoDB Atlas persistence layer, and Bedrock-backed
  LLM calls.
- `services/prompting.py` – shared prompt and context-normalisation helpers for
  both the CLI demo and AgentCore tooling.
- `agent/agentcore_config.py` – declarative AgentCore runtime, memory, identity
  and tool configuration for the retrieval-augmented pipeline.
- `agent/tool_definitions.json` – Model Context Protocol (MCP) tool definitions
  for the S3 ingest, MongoDB Atlas search, and Bedrock answer tools.
- `streamlit_app.py` – chat UI with video-aware filtering, citations, and
  optional Bedrock streaming support.

## Requirements

- Python 3.8 or later
- `boto3`
- `pymongo`
- `scikit-learn`
- `numpy`
- `voyageai` *(optional but recommended for YouTube embeddings; install when
  using YouTubeRAG-generated vectors).*

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
- `ATLAS_COLLECTION_NAME` (default: `video_chunks`)
- `ATLAS_SEARCH_INDEX_NAME` (default: `video_chunks_rag_index`)
- `ATLAS_EMBEDDING_DIM` (default: `1024`): must match the embedding length used
  by YouTubeRAG or any custom ingestion pipeline.
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
- `VOYAGE_API_KEY` *(optional)*: enables Voyage embeddings (`voyage-large-2`)
  for both ingestion and query, aligning with the default YouTubeRAG pipeline.
- `VOYAGE_MODEL` *(optional)*: override the Voyage embedding model name.

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
   issues an Atlas vector search, routes the retrieved YouTube segments into an
   Amazon Bedrock model, and returns a grounded answer with citations that
   highlight the video title, timestamp, speaker, and channel.

4. Type `exit` (or press `Ctrl+C`) to quit.

## YouTubeRAG media pipeline

To ingest rich YouTube metadata, transcripts, and embeddings:

1. Clone or reference the [YouTubeRAG](https://github.com/FernandoBDAF/YoutubeRAG) repository.
2. Create the MongoDB collections and Atlas Search indexes defined in
   `mongodb_schema.json`. Ensure the `video_chunks` collection and its vector
   index use 1024 dimensions.
3. Export the required environment variables (including `MONGODB_ATLAS_URI`,
   `MONGODB_DB`, `VOYAGE_API_KEY`, AWS Bedrock credentials, etc.).
4. Run the YouTubeRAG pipeline to process a playlist:

   ```bash
   python main.py pipeline --playlist_id <YOUTUBE_PLAYLIST_ID> --max 5 --llm
   ```

   This will populate the `video_chunks`, `memory_logs`, and related collections
   with diarised transcript segments, enriched metadata, and Voyage embeddings.
5. Back in this repository, set `ATLAS_COLLECTION_NAME=video_chunks`,
   `ATLAS_SEARCH_INDEX_NAME=video_chunks_rag_index`, and `ATLAS_EMBEDDING_DIM=1024`
   (defaults already match these values). The agent now queries video segments
   directly from Atlas using the same embedding space as YouTubeRAG.

Optional filters exposed in the agent (`channel`, `speaker`, `video_id`,
`playlist_id`, `start_time`, `end_time`) map to metadata fields produced by
YouTubeRAG, enabling targeted searches such as “When does the CEO mention
sustainability?” or “Find segments where the speaker covers monetisation.”

## Streamlit Chat Interface

For a browser-based experience run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The interface reads the same environment variables as `demo.py` and exposes:
- Live chat with short-term memory and media-aware citations (video title,
  channel, speaker, timestamp, and watch links).
- Sidebar controls for retrieval depth, response length, temperature, and streaming.
- YouTube-specific filters (channel, speaker, video, playlist, timestamp) applied
  directly to Atlas vector search queries.
- Optional S3 ingestion so you can refresh the MongoDB Atlas knowledge base without leaving the page.
- Session reset controls to clear the conversation without restarting the app.

Run the command from the repository root after installing the dependencies in `requirements.txt`.

## Deploying with Bedrock AgentCore

This repository now includes everything required to host the retrieval pipeline
as a Bedrock AgentCore agent. The `agent/agentcore_config.py` module exposes a
single `build_agentcore_configuration` function that produces a complete
runtime/memory/identity/tool configuration compatible with AgentCore's SDK. The
same module also implements the three gateway tools referenced in the proposal:

- `s3_ingest_tool` uploads fresh content from Amazon S3, chunks and embeds the
  text, and upserts the vectors into MongoDB Atlas.
- `mongo_search_tool` performs a vector Atlas Search query using the same embedding
  model configured for ingestion (Voyage or hashing). Optional filters allow the
  agent to target specific channels, speakers, playlists, or time ranges.
- `bedrock_answer_tool` accepts retrieved context, builds a grounded prompt via
  `services.prompting.build_grounded_answer_prompt`, and calls the configured
  Amazon Bedrock model. The prompt now references YouTube segment metadata so the
  generated answer can cite titles, timestamps, and watch links.

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

- When `VOYAGE_API_KEY` is present the agent uses Voyage embeddings (default
  `voyage-large-2`) to remain compatible with YouTubeRAG. Without the API key,
  the code falls back to deterministic `HashingVectorizer` embeddings.
- Ensure the Atlas Search index definition matches the configured embedding
  dimensionality (default 1024 for media scenarios).
- The Atlas index definition is created on startup if it does not already
  exist. Re-running the script will upsert chunks by a deterministic chunk ID
  (`<s3-key>:::<chunk-index>`).
- Answers are generated by the configured Bedrock model. Provide AWS
  credentials through `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and, when
  applicable, `AWS_SESSION_TOKEN`.
