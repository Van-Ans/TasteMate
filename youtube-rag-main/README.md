# # Tastemate AI

AI-powered educational YouTube RAG assistant using LangChain, Ollama, and vector search.

## Features
- YouTube lecture transcript extraction
- AI-powered chatbot
- Educational content summarization
- Related lecture recommendations
- Vector search using embeddings
- Personalized learning assistant

## Tech Stack
- Python
- LangChain
- FastAPI
- Pinecone/ChromaDB
- OpenAI API
- YouTube API

## Required Environment Variables
Create a `.env` file in the project root with the following values:

- `YOUTUBE_USER_HANDLE` - the YouTube channel handle to index/retrieve from
- `YOUTUBE_API_KEY` - a valid Google YouTube Data API key
- `OPENAI_KEY` - your OpenAI API key
- `PINECONE_API_KEY` - your Pinecone API key
- `PINECONE_ENVIRONMENT` - your Pinecone environment string (for example `us-west1-gcp`)
- `PINECONE_INDEX_NAME` - optional, defaults to `youtube-videos`

## Run the app
1. Install dependencies:
   ```bash
   python -m poetry install --no-root
   ```
2. Populate your Pinecone index by running the indexing pipeline or using the existing Dagster assets.
3. Start the FastAPI server:
   ```bash
   python -m uvicorn src.serving.serve:app --reload
   ```
