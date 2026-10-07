# Document Q&A Chatbot with RAG

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a PDF document. Instead of relying only on what the LLM already knows, it retrieves the most relevant chunks of the document and gives them to the LLM as context, so answers are grounded in the source.

Built with **LangChain**, **LangGraph**, **FAISS**, **Google Gemini** and **Streamlit**.

## Architecture

```
PDF -> extract text -> remove references -> split into chunks
    -> Gemini embeddings -> FAISS index (saved to disk)

Question -> retrieve top-k chunks -> generate answer -> grade -> answer
                  ^                                        |
                  |------ not grounded: retry (more chunks)
```

LangGraph state graph:

1. **retrieve**: similarity search in FAISS (top 3 chunks, 5 on retry).
2. **generate**: Gemini answers using only the retrieved context.
3. **grade**: Gemini checks whether the answer is supported by the context (optional, can be switched off to save API quota).
4. **decide**: if the answer is not grounded, go back to retrieve with more chunks (maximum 2 attempts).

If the Gemini API is busy or the quota is finished, the app does not crash. It shows the most relevant passages from the document instead.

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3 |
| LLM | Google Gemini (`gemini-3.8-flash`) |
| Embeddings | Google `gemini-embedding-001` |
| Vector store | FAISS (local) |
| Orchestration | LangChain + LangGraph |
| PDF reading | PyPDF2 |
| Interface | Streamlit |

## Project structure

```
RAG-Chatbot/
├── APP/
│   └── streamlit_app.py        # Chat UI (LangGraph + saved index + fallback)
├── data/
│   └── documents/              # Source PDF
├── vectorstore/
│   └── faiss_index/            # Saved index (generated, not committed)
├── rag_experiments.ipynb       # Ingestion, chunking and retrieval experiments
├── .env.example                # Template for the API key
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the project and create a virtual environment**

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Add your API key**

   Get a key from https://aistudio.google.com/apikey, then copy `.env.example` to `.env` and edit it:

   ```
   GOOGLE_API_KEY=your_key_here
   ```

   Never commit the real `.env` file.

3. **Build the vector index (one time)**

   Open `rag_experiments.ipynb`, select the `.venv` kernel and run cells 1 to 11 from top to bottom. Cell 7 creates `vectorstore/faiss_index`.

4. **Run the app**

   ```powershell
   streamlit run APP\streamlit_app.py
   ```

## Retrieval experiments

- The PDF text contained a long References section (links, DOI numbers). It was removed, which reduced the text from 17,944 to 11,122 characters.
- `CharacterTextSplitter` (1000 characters) mixed unrelated headings in one chunk. `RecursiveCharacterTextSplitter` (800 characters, 150 overlap) was used instead, giving 17 clean chunks.
- `k = 3` gave the most relevant context. Higher values added chunks that were only loosely related.

## Example Q&A

_Replace these with your own screenshots and answers._

**Q: What is RAG?**
A: Retrieval-augmented generation is a technique that enables large language models to retrieve and incorporate new information from external data sources before generating responses.

**Q: Why do we need RAG?**
A: It reduces hallucinations, gives access to updated or domain-specific data, saves the cost of retraining, and lets the model cite sources.

## Screenshots

_Add screenshots of the chat UI here._

## Notes and limitations

- The free Gemini tier has a small daily request limit. `USE_GRADING = False` in `streamlit_app.py` uses one LLM call per question instead of two.
- The chatbot answers only from the loaded document and says "I don't know" otherwise.
- Each question is answered independently (no conversation memory yet).

## Possible improvements

- Conversation memory for follow-up questions
- Upload your own PDFs from the UI
- Streaming answers
- Unit tests for retrieval