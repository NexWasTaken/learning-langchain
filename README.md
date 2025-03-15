# Learning Langchain

Examples with langchain following the tutorial @ [LangChain Master Class For Beginners 2024 [+20 Examples, LangChain V0.2]](https://www.youtube.com/watch?v=yF9kGESAi3M)

I skipped the ones that I felt were redundant.

## Topic Covered

### 1. Chat Models

### 2. Prompt Templates

### 3. Chains

### 4. RAG

- Storing a document/knowledge base:

  - Split the document into chunks.
  - Create embeddings for every chunk.
    - ℹ️ What are embeddings?
    - Simply numerical representation of text.
  - Store embeddings in vector database or store.

- Querying the document/knowledge base:

  - Convert the query into vector embeddings.
  - Call the DB Retriever to match the query embeddings vs the document chunk embeddings.

- Add metadata for better answers.

### 5. Agents & Tools

Three states: Action, Observation, Thought
