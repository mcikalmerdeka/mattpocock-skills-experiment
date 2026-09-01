# Doc QA

A question-answering application: users upload documents, and the model answers conversational questions by retrieving relevant document content from a vector store.

## Language

### Ingestion

**Document**:
A user-supplied TXT or Markdown file uploaded through the app to be made answerable.
_Avoid_: file, upload, source

**Chunk**:
A split piece of a Document that is embedded and retrieved as one unit.
_Avoid_: segment, passage, slice

**Ingestion**:
The act of turning a Document into embedded Chunks stored in the Vector Store.
_Avoid_: indexing, importing, parsing

**Vector Store**:
The persistent local store holding embedded Chunks of every ingested Document.
_Avoid_: database, index, collection

### Conversation

**Conversation**:
A named chat session between the user and the model, with its own retained history.
_Avoid_: session, thread, chat

**Query**:
The user's message within a Conversation that triggers Retrieval and an Answer.
_Avoid_: prompt, question

**Retrieval**:
Selecting the most relevant Chunks from the Vector Store for a given Query.
_Avoid_: search, lookup, fetching

**Answer**:
The model's reply to a Query, grounded in Retrieved Chunks and Conversation History.
_Avoid_: response, output
