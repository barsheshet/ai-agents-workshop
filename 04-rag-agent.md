# 04 - Building a RAG Agent with Evaluator-Optimizer Pattern

This guide walks you through building a **RAG (Retrieval Augmented Generation)** agent that answers questions from your documents. It also demonstrates the **Evaluator-Optimizer** pattern for self-correction.

> **Prerequisites:** Complete [00-environment-setup.md](./00-environment-setup.md) first.

Install dependencies:

```bash
uv add langchain-openai langgraph langchain-community langchain-text-splitters python-dotenv
```

---

## What is RAG?

RAG (Retrieval Augmented Generation) is a technique where:

1. **Retrieve** - Find relevant documents based on the user's question
2. **Augment** - Add those documents to the LLM's context
3. **Generate** - LLM generates an answer using the retrieved context

This allows the LLM to answer questions about your specific documents (wiki, docs, etc.) that it wasn't trained on.

---

## What is Evaluator-Optimizer?

An agentic pattern where the agent checks its own work:

- **Evaluator**: Checks if the retrieved documents are relevant to the question
- **Optimizer**: If not relevant, rewrites the query and tries again

This creates a self-correcting loop that improves answer quality.

---

## Step 1: Import Libraries

```python
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
```

**Key imports:**

- `WebBaseLoader` - Loads content from web pages
- `RecursiveCharacterTextSplitter` - Splits documents into chunks
- `InMemoryVectorStore` - Stores document embeddings for retrieval
- `OpenAIEmbeddings` - Creates vector embeddings from text

---

## Step 2: Load Environment Variables

```python
load_dotenv()

API_KEY = os.environ.get("API_KEY", "")
API_URL = "https://truefoundry.staging.sunbit.in/api/llm"
MODEL = "vertex-staging/gemini-2-5-flash"
EMBEDDING_MODEL = "vertex-staging/gemini-embedding-001"
```

---

## Step 3: Load and Process Documents

Load documents from URLs and split them into chunks for embedding.

```python
print("[1/5] Loading documents...")

urls = [
    "https://example.com/docs/page1",
    "https://example.com/docs/page2",
    # Add your document URLs here
]

docs = []
for url in urls:
    try:
        docs.extend(WebBaseLoader(url).load())
    except Exception as e:
        print(f"  Warning: Could not load {url}: {e}")

print(f"  Loaded {len(docs)} documents")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"  Split into {len(chunks)} chunks")
```

**Key Points:**

- `chunk_size=500` - Each chunk is ~500 characters
- `chunk_overlap=50` - Chunks overlap to preserve context at boundaries
- Smaller chunks = more precise retrieval, but less context per chunk

---

## Step 4: Create the Vector Store

Create embeddings for each chunk and store them in a vector database.

```python
print("[2/5] Creating vector store...")

llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=API_URL)
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=API_KEY,
    base_url=API_URL,
    check_embedding_ctx_length=False,
)

vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()
print("  Vector store ready")
```

**Key Points:**

- `InMemoryVectorStore` - Simple in-memory storage (for production, use FAISS, Pinecone, etc.)
- `as_retriever()` - Creates a retriever interface for searching documents
- The retriever returns the most similar documents to a query

---

## Step 5: Define the Retriever Tool

Create a tool that the agent can use to search documents.

```python
@tool
def search_docs(query: str) -> str:
    """Search the documentation for relevant information."""
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

tools = [search_docs]
```

**Key Points:**

- The `@tool` decorator makes this function available to the LLM
- The docstring becomes the tool description (helps LLM understand when to use it)
- Returns retrieved document content as a single string

---

## Step 6: Define the State

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

Same as before - `add_messages` handles appending messages automatically.

---

## Step 7: Define Helper Function

A helper to extract retrieved documents from the message history.

```python
def get_docs_from_messages(messages: list) -> str:
    """Extract retrieved docs from message history."""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "tool":
            return msg.content
    return ""
```

---

## Step 8: Define the Agent Node

The agent decides whether to search for documents or respond directly.

```python
def agent(state: State):
    """LLM decides whether to search docs or respond directly."""
    print("  [Agent] Deciding next action...")
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}
```

---

## Step 9: Define the Evaluator Node

The **Evaluator** checks if retrieved documents are relevant to the question.

```python
def evaluate(state: State):
    """
    EVALUATOR: Check if retrieved docs are relevant.
    Returns 'yes' or 'no'.
    """
    print("  [Evaluate] Checking if docs are relevant...")

    question = state["messages"][0].content
    docs = get_docs_from_messages(state["messages"])

    prompt = f"""Are these documents relevant to the question? Answer only 'yes' or 'no'.

Question: {question}

Documents:
{docs[:1000]}...

Relevant (yes/no):"""

    response = llm.invoke(prompt)
    return {"messages": [response]}
```

**Key Points:**

- Takes the original question and retrieved documents
- Asks the LLM to judge relevance
- Simple yes/no output for easy routing

---

## Step 10: Define the Optimizer Node

The **Optimizer** rewrites the query when documents aren't relevant.

```python
def rewrite(state: State):
    """
    OPTIMIZER: Rewrite the query for better search results.
    """
    print("  [Rewrite] Improving the query...")

    original_question = state["messages"][0].content

    prompt = f"""Look at the input and try to reason about the underlying semantic intent / meaning.

            Here is the initial question:
            -------
            {original_question}
            -------

            Formulate an improved question. Do not add any other text or explanation. Just the question."""

    response = llm.invoke(prompt)
    print(f"  [Rewrite] New query: {response.content}")
    # Return as HumanMessage so agent will search with new query
    return {"messages": [HumanMessage(content=response.content)]}
```

**Key Points:**

- Creates a better version of the original question
- Returns as `HumanMessage` so the agent will process it as a new query
- Loops back to the agent node for another search attempt

---

## Step 11: Define the Generate Node

Creates the final answer using the retrieved documents.

```python
def generate(state: State):
    """Generate the final answer using retrieved documents."""
    print("  [Generate] Creating response...")

    question = state["messages"][0].content
    docs = get_docs_from_messages(state["messages"])

    prompt = f"""Answer the question using the context. Be concise (3 sentences max).

Question: {question}

Context:
{docs}

Answer:"""

    response = llm.invoke(prompt)
    return {"messages": [response]}
```

---

## Step 12: Define the Routing Logic

Routes based on the evaluator's decision.

```python
def check_relevance(state: State) -> str:
    """Route: generate if relevant, rewrite if not."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "content"):
        if "yes" in last_message.content.lower():
            print("  [Router] Docs relevant → generate")
            return "generate"

    print("  [Router] Docs not relevant → rewrite")
    return "rewrite"
```

---

## Step 13: Build the Graph

Connect all nodes with edges to create the workflow.

```python
print("[4/5] Building agent graph...")

graph = StateGraph(State)

# Add all nodes
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_node("evaluate", evaluate)      # EVALUATOR
graph.add_node("rewrite", rewrite)        # OPTIMIZER
graph.add_node("generate", generate)

# Define the flow
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # → tools or END
graph.add_edge("tools", "evaluate")
graph.add_conditional_edges("evaluate", check_relevance, ["generate", "rewrite"])
graph.add_edge("rewrite", "agent")  # Try again with rewritten query
graph.add_edge("generate", END)

rag_agent = graph.compile()
print("[5/5] Agent ready!")
```

**Key Points:**

- `tools_condition` - Prebuilt routing: goes to "tools" if agent wants to search, else END
- `check_relevance` - Custom routing: goes to "generate" if docs relevant, else "rewrite"
- The rewrite node loops back to agent for another search attempt

---

## Step 14: Run the Agent

```python
def ask(question: str):
    """Ask the RAG agent a question."""
    print("=" * 60)
    print(f"Q: {question}")
    print("=" * 60)

    result = rag_agent.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content

    print(f"\nA: {answer}\n")


if __name__ == "__main__":
    if not API_KEY:
        print("Error: Set API_KEY in your .env file")
    else:
        ask("How do I connect to the database?")
```

Run it:

```bash
uv run 04-rag-agent.py
```

---

## How It Works (Flow Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                        USER QUESTION                         │
│              "How do I connect to the database?"             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        ┌───────────┐
                        │   agent   │  ◀───────────────────────┐
                        └───────────┘                          │
                              │                                │
                              ▼                                │
                    ┌─────────────────┐                        │
                    │ tools_condition │                        │
                    └─────────────────┘                        │
                     /              \                          │
              search?                no search                 │
                   /                      \                    │
                  ▼                        ▼                   │
            ┌───────────┐            ┌───────────┐             │
            │   tools   │            │    END    │             │
            │ (search)  │            └───────────┘             │
            └───────────┘                                      │
                  │                                            │
                  ▼                                            │
            ┌───────────┐                                      │
            │ evaluate  │  ← EVALUATOR: Are docs relevant?     │
            └───────────┘                                      │
                  │                                            │
                  ▼                                            │
         ┌───────────────┐                                     │
         │check_relevance│                                     │
         └───────────────┘                                     │
          /            \                                       │
       yes              no                                     │
        /                \                                     │
       ▼                  ▼                                    │
┌───────────┐       ┌───────────┐                              │
│ generate  │       │  rewrite  │  ← OPTIMIZER: Better query   │
└───────────┘       └───────────┘                              │
      │                   │                                    │
      ▼                   └────────────────────────────────────┘
┌───────────┐
│    END    │
└───────────┘
```

---

## The Evaluator-Optimizer Pattern

This pattern creates a self-correcting agent:

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATOR-OPTIMIZER LOOP                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. Agent retrieves documents                               │
│                    ↓                                         │
│   2. EVALUATOR: "Are these docs relevant?"                   │
│                    ↓                                         │
│         ┌─────────┴─────────┐                                │
│         │                   │                                │
│        YES                  NO                               │
│         │                   │                                │
│         ▼                   ▼                                │
│   3. Generate          4. OPTIMIZER: Rewrite query           │
│      answer                 │                                │
│                             └──────→ Back to step 1          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Why this matters:**

- First search might miss relevant docs due to poor query phrasing
- The optimizer reformulates the question with better keywords
- This mimics how humans refine their search when they don't find what they need

---

## Key Concepts Summary

| Concept          | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| **RAG**          | Retrieve docs → Add to context → Generate answer                   |
| **Vector Store** | Database that stores document embeddings for semantic search       |
| **Embeddings**   | Numerical representations of text that capture meaning             |
| **Chunking**     | Splitting documents into smaller pieces for better retrieval       |
| **Evaluator**    | Node that checks if the current step's output is good enough       |
| **Optimizer**    | Node that improves the input when evaluator says "not good enough" |
