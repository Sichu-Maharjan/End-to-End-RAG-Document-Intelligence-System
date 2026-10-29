# End-to-End RAG + Document Intelligence System

Chat with a collection of research PDFs. Hybrid search (semantic + BM25),
cross-encoder reranking, cited answers from a local LLM, login, document upload,
chat history, monitoring and an evaluation dashboard.

## Architecture
- **Backend:** FastAPI (`app/main.py`, `app/rag_pipeline.py`)
- **Frontend:** Streamlit (`app/streamlit_app.py`, `app/pages/`)
- **Vector DB:** ChromaDB, plus BM25 keyword index
- **LLM:** llama3.2:3b via Ollama (runs on the host)
- **Storage:** SQLite for chat history and document records

## Requirements
- Docker Desktop
- Ollama, with the model pulled: `ollama pull llama3.2:3b`

## Data setup
The data folders (`PDFs/`, `data/`, `processed/`) are not in the repository
because they are large. To run the project, regenerate them by running the
notebook cells for Steps 2-9 (`notebook/intelligenceSystem.ipynb`), then run
`docker compose up --build`.

## Run
1. Start Ollama.
2. Put the data in `processed/` and `PDFs/` (see the notebook, Steps 2-9, to generate them).
3. From this folder:

       docker compose up --build

4. Open http://localhost:8501 and log in.

## Notes
- First start is slow: models download inside the container.
- Default login is defined in `app/auth.py`
- Data folders are gitignored, so they are not in the repository.