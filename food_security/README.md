# Food Security Recommender (RAG)

This repository contains a Jupyter notebook that implements a Retrieval-Augmented Generation (RAG) system to provide evidence-based recommendations for food security questions. The notebook loads research documents (PDFs), creates semantic embeddings, stores them in a vector store, and uses an LLM to generate grounded answers.

## Notebook: food_security.ipynb

Purpose: Ground LLM output in domain-specific documents about food security so recommendations are supported by source material.

High-level pipeline:

1. Load PDFs from `data/` and split into smaller chunks.
2. Create embeddings for chunks using an embedding model (OpenAI or compatible API).
3. Store embeddings in an in-memory vector store for semantic similarity search.
4. Retrieve top-k chunks for a user query and condition an LLM prompt on those chunks.
5. Generate actionable, context-aware recommendations.

See `notebook_description.txt` for a detailed walkthrough of the notebook steps and screenshot placeholders.

## Quick start

- Install dependencies (create a virtual environment, then install project dependencies):

	# Use your environment's package manager; example for pip:
	# python -m venv .venv
	# source .venv/bin/activate
	# pip install -r requirements.txt

- Add your API key in a `.env` file or as environment variables:

	- For OpenAI: set `OPENAI_API_KEY`
	- For OpenRouter (optional): set `OPENROUTER_API_KEY` and optionally `OPENROUTER_API_BASE`

- Run the notebook cells in order: 1 → 2 → 3 → 4 → 5 → 6

## Troubleshooting

- AuthenticationError (401) when creating embeddings: check that `OPENAI_API_KEY` or `OPENROUTER_API_KEY` is valid and not truncated or rotated. Do not commit secrets to the repo.
- If PDFs are not found, place them in the `data/` folder or update the path used by the notebook.

## Security

- Never commit API keys or other secrets. Use `.env` and add it to `.gitignore`.

## Files

- `food_security.ipynb` — the main notebook (RAG pipeline).
- `notebook_description.txt` — a human-readable walkthrough and screenshot placeholders.
- `data/` — place your PDF documents here.

---

If you'd like, I can also:

- Insert the notebook description into the notebook as a markdown cell.
- Generate a `requirements.txt` or update `pyproject.toml` with required packages.


