# Doc QA

A local, ChatGPT-style web app: upload TXT or Markdown documents, then chat with a model that answers from *your* material — grounded in the content of the documents you've ingested. Conversations are saved, so you can pick any past conversation back up.

Model calls, embeddings, and the local vector store all run through [LangChain](https://docs.langchain.com/) — `langchain-openai` (ChatOpenAI, OpenAIEmbeddings), `langchain-chroma`, and `langchain-text-splitters`. No raw OpenAI or Chroma clients are used anywhere in the app.

## Run it

1. Copy `.env.example` to `.env` and set your `OPENAI_API_KEY`.
2. Start the app with a single command:

```sh
uv run doc-qa
```

Streamlit opens the Doc QA page in your browser.

## Tests

```sh
uv run pytest
```
