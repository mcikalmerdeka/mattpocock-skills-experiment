# Doc QA

A local, ChatGPT-style web app: upload TXT or Markdown documents, then chat with a model that answers from *your* material — grounded in the content of the documents you've ingested. Conversations are saved, so you can pick any past conversation back up.

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
