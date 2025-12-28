# 03 - Building a RAG Agent with LangGraph

This guide walks you through building a **Retrieval-Augmented Generation (RAG)** agent using LangGraph. The agent searches Sunbit's Confluence wiki to answer questions, with built-in evaluation and query rewriting for better results.

## How It Works (Flow Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│           "How to run a service in IntelliJ?"               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        ┌───────────┐
              ┌────────▶│   agent   │◀────────────────────────┐
              │         │   (LLM)   │                         │
              │         └───────────┘                         │
              │               │                               │
              │               ▼                               │
              │     ┌─────────────────┐                       │
              │     │   tool calls?   │                       │
              │     └─────────────────┘                       │
              │      /              \                         │
              │    YES               NO                       │
              │    /                  \                       │
              │   ▼                    ▼                      │
              │ ┌───────────┐    ┌───────────┐                │
              │ │   tools   │    │    END    │                │
              │ │ (search)  │    └───────────┘                │
              │ └───────────┘                                 │
              │       │                                       │
              │       ▼                                       │
              │ ┌───────────┐                                 │
              │ │ evaluate  │  ← Are docs relevant?           │
              │ └───────────┘                                 │
              │    /      \                                   │
              │  YES       NO                                 │
              │  /          \                                 │
              │ ▼            ▼                                │
        ┌───────────┐  ┌───────────┐                          │
        │ generate  │  │  rewrite  │──────────────────────────┘
        │  answer   │  │   query   │
        └───────────┘  └───────────┘
              │
              ▼
        ┌───────────┐
        │    END    │
        └───────────┘
```

**Key Concepts:**

- **Evaluator**: Checks if retrieved documents are relevant
- **Optimizer**: Rewrites the query if documents aren't relevant
- **Self-correction loop**: Agent retries with improved queries

---

> **Prerequisites:** Complete [00-environment-setup.md](./00-environment-setup.md) first.

Install dependencies:

```bash
uv add langchain-openai langchain-community langchain-text-splitters langgraph python-dotenv beautifulsoup4
```

---

## Step 0a: Create the Python File

Create a new file called `03-rag-agent.py`:

```bash
touch 03-rag-agent.py
```

Open the file in Cursor and follow the steps below.

---

## Step 0b: Configure Confluence Cookie

To access Sunbit's Confluence wiki, you need to extract the session cookie:

1. Open [Sunbit Confluence](https://sunbit.atlassian.net/wiki) in your browser
2. Open Developer Tools (`F12` or `Cmd + Option + I`)
3. Go to the **Application** tab → **Cookies** → `https://sunbit.atlassian.net`
4. Find the cookie named `tenant.session.token`
5. Copy its value

Add it to your `.env` file:

```
API_KEY=your_api_key_here
CONFLUENCE_COOKIE=<paste_your_token_here>
```

---

## Step 1: Import Libraries

```python
# 1. import libraries
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

- `WebBaseLoader` - Loads documents from web pages
- `RecursiveCharacterTextSplitter` - Splits documents into chunks
- `InMemoryVectorStore` - Simple vector store for embeddings
- `OpenAIEmbeddings` - Embedding model client
- `StateGraph` - LangGraph's graph builder
- `ToolNode`, `tools_condition` - Prebuilt components for tool handling

---

## Step 2: Load Environment Variables

```python
# 2. load environment variables
load_dotenv()
```

---

## Step 3: Setup API Configuration

```python
# 3. setup API configuration
API_KEY = os.environ.get("API_KEY", "")
CONFLUENCE_COOKIE = os.environ.get("CONFLUENCE_COOKIE", "")
print(f"CONFLUENCE_COOKIE: {CONFLUENCE_COOKIE}")
API_URL = "https://truefoundry.staging.sunbit.in/api/llm"  # API URL
MODEL = "vertex-staging/gemini-2-5-flash"  # LLM
EMBEDDING_MODEL = "vertex-staging/gemini-embedding-001"  # Embeddings
```

**Note:** You need a `CONFLUENCE_COOKIE` to access Sunbit's wiki pages.

---

## Step 4: Load and Process Documents

Load documents from Confluence URLs and split them into chunks for embedding.

```python
# 4. load and process documents
print("[1/5] Loading documents...")

urls = [
    "https://sunbit.atlassian.net/wiki/spaces/DEV/pages/2322956298/How+To+connect+Sunbit+s+applicative+DB",  # How To connect Sunbit's applicative DB
    "https://sunbit.atlassian.net/wiki/spaces/DEV/pages/2567864415/How-To-Collaborate+on+Local+Network",  # How-To-Collaborate on Local Network
    "https://sunbit.atlassian.net/wiki/spaces/DEV/pages/2544404403/Step-by-Step+Guide+to+Running+Service+in+IntelliJ+IDEA+Community+Edition+CE",  # Step-by-Step Guide to Running Service in IntelliJ IDEA Community Edition (CE)
    "https://sunbit.atlassian.net/wiki/spaces/DEV/pages/2413363251/How+to+Connect+to+Remote-DEV",  # How to Connect to Remote-DEV
    "https://sunbit.atlassian.net/wiki/spaces/DEV/pages/2593587645/New+RDS+access+scheme",  # How to requst access to RDS
    "https://sunbit.atlassian.net/wiki/spaces/DEV/pages/2651784449/How+to+check+vulnerabilities+in+you+project",  # How to check vulnerabilities in you project
    "https://sunbit.atlassian.net/wiki/spaces/DEV/pages/2396815615/How+to+add+Metrics+with+Prometheus+to+a+Service",  # how to add metrics with prometheus to a service
    "https://sunbit.atlassian.net/wiki/spaces/DEV/pages/2292154407/How+to+Create+a+Monitor+in+DataDog",  # how to create a monitor in datadog
]

docs = []
for url in urls:
    try:
        docs.extend(
            WebBaseLoader(
                url,
                requests_kwargs={
                    "headers": {"Cookie": f"tenant.session.token={CONFLUENCE_COOKIE}"}
                },
            ).load()
        )
    except Exception as e:
        print(f"  Warning: Could not load {url}: {e}")

print(f"  Loaded {len(docs)} documents")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"  Split into {len(chunks)} chunks")
```

**Key Points:**

- Uses `WebBaseLoader` with authentication cookie
- Splits documents into 500-character chunks with 50-character overlap
- Handles loading errors gracefully

---

## Step 5: Create Vector Store

Create embeddings and store them in an in-memory vector store.

```python
# 5. create vector store
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

---

## Step 6: Define the Retriever Tool

Create a tool that searches the vector store.

```python
# 6. define the retriever tool
@tool
def search_docs(query: str) -> str:
    """Search the Sunbit wiki for relevant information."""
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


tools = [search_docs]
```

---

## Step 7: Define the Graph State

Define the state that flows through the graph.

```python
# 7. define the graph state and nodes
print("[3/5] Setting up agent nodes...")


class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**Note:** `add_messages` is a reducer that appends new messages to the list instead of replacing it.

---

## Step 8: Define the Helper Function

A utility to extract retrieved documents from the message history.

```python
# 8. define the helper function
# This function extracts the retrieved documents from the message history.
def get_docs_from_messages(messages: list) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "tool":
            return msg.content
    return ""
```

---

## Step 9: Define the Agent Node

The main LLM node that decides whether to search or respond.

```python
# 9. define the agent node
# This node decides whether to search or respond directly.
def agent(state: State):
    """LLM decides whether to search docs or respond directly."""
    print("  [Agent] Deciding next action...")
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}
```

---

## Step 10: Define the Evaluator Node

Checks if the retrieved documents are relevant to the question.

```python
# 10. define the evaluator node
# This node checks if the retrieved documents are relevant to the question.
def evaluate(state: State):
    """
    EVALUATOR: Check if retrieved docs are relevant to the question.
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

**Key Point:** This is the **evaluator** pattern - the LLM judges the quality of retrieved results.

---

## Step 11: Define the Optimizer Node

Rewrites the query if the documents aren't relevant.

```python
# 11. define the optimizer node
# This node rewrites the query to get better search results.
def rewrite(state: State):
    """
    OPTIMIZER: Rewrite the query to get better search results.
    This creates a new, improved question.
    """
    print("  [Rewrite] Improving the query...")

    original_question = state["messages"][0].content

    prompt = f"""Look at the input and try to reason about the underlying semantic intent / meaning.

                Here is the initial question:

                -------

                {original_question}

                -------

                Formulate an improved question. do not add any other text or explanation. Just the question."""

    response = llm.invoke(prompt)
    rewritten_query = response.content
    print(f"  [Rewrite] New query: {rewritten_query}")
    # Return as HumanMessage so the agent will search with the new query
    return {"messages": [HumanMessage(content=rewritten_query)]}
```

**Key Point:** This is the **optimizer** pattern - improving inputs to get better results.

---

## Step 12: Define the Generate Node

Creates the final answer using the retrieved documents.

```python
# 12. define the generate node
# This node creates the final answer using the retrieved documents.
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

## Step 13: Define the Routing Function

Routes the flow based on the evaluator's decision.

```python
# 13. define the routing node
# This node routes the flow based on the evaluator's decision.
def check_relevance(state: State) -> str:
    """Route based on evaluator's decision: generate or rewrite."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "content"):
        if "yes" in last_message.content.lower():
            print("  [Router] Docs relevant → generate")
            return "generate"

    print("  [Router] Docs not relevant → rewrite")
    return "rewrite"
```

---

## Step 14: Build the Graph

Assemble all nodes and edges into a compiled graph.

```python
# 14. build the graph
# This function builds the graph of nodes and edges.
print("[4/5] Building agent graph...")

graph = StateGraph(State)

# Add all nodes
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_node("evaluate", evaluate)  # Evaluator Node
graph.add_node("rewrite", rewrite)  # Optimizer Node
graph.add_node("generate", generate)

# Define the flow
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # → tools or END
graph.add_edge("tools", "evaluate")
graph.add_conditional_edges("evaluate", check_relevance, ["generate", "rewrite"])
graph.add_edge("rewrite", "agent")  # Try again with rewritten query
graph.add_edge("generate", END)

rag_agent = graph.compile()
print("[5/5] Agent ready!\n")
```

**Graph Flow:**

1. `START` → `agent`: LLM decides to search
2. `agent` → `tools`: Execute the search
3. `tools` → `evaluate`: Check document relevance
4. `evaluate` → `generate` OR `rewrite`: Based on relevance
5. `rewrite` → `agent`: Loop back with improved query
6. `generate` → `END`: Return final answer

---

## Step 15: Run the Agent

```python
# 15. run the agent
# This function runs the agent with a given question.
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
        ask("how to run a service in IntelliJ?")
```

Run it:

```bash
uv run 03-rag-agent.py
```
