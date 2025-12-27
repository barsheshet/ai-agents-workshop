# 02 - Building an AI Agent with create_agent: Infra Doctor

This guide walks you through building the same **Site Reliability Engineer (SRE)** agent using LangChain's `create_agent` - the simplest approach with minimal code.

## How It Works (Flow Diagram)

`create_agent` builds this graph internally:

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│        "Investigate performance issues on prod-db-01"       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        ┌───────────┐
                        │   agent   │  ◀──────────────┐
                        │   (LLM)   │                 │
                        └───────────┘                 │
                              │                       │
                              ▼                       │
                    ┌─────────────────┐               │
                    │   tool calls?   │               │
                    └─────────────────┘               │
                     /              \                 │
                   YES               NO               │
                   /                  \               │
                  ▼                    ▼              │
            ┌───────────┐        ┌───────────┐       │
            │   tools   │        │    END    │       │
            └───────────┘        └───────────┘       │
                  │                                   │
                  └───────────────────────────────────┘
```

---

> **Prerequisites:** Complete [00-environment-setup.md](./00-environment-setup.md) first.

Install dependencies:

```bash
uv add langchain-openai langchain python-dotenv
```

---

## Step 1: Import Libraries

```python
# --- 1. Import Libraries ---
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
```

**Key imports:**

- `ChatOpenAI` - LLM client compatible with OpenAI API
- `@tool` - Decorator to define tools with automatic schema generation
- `create_agent` - Prebuilt agent that handles the entire ReAct loop

---

## Step 2: Load Environment Variables

```python
# --- 2. Load Environment Variables ---
load_dotenv()
```

---

## Step 3: Setup API Configuration

```python
# --- 3. Setup API Configuration ---
API_KEY = os.environ.get("API_KEY", "")
API_URL = "https://truefoundry.staging.sunbit.in/api/llm"
MODEL = "vertex-staging/gemini-2-5-flash"
```

---

## Step 4: Define the Tools

Use the `@tool` decorator to define tools. The decorator automatically generates the JSON schema from the function signature and docstring.

```python
# --- 4. Define the Tools ---
@tool
def check_cpu_usage(server_id: str) -> str:
    """Checks real-time CPU usage for a specific server."""
    print(f"  [TOOL] Checking CPU for {server_id}...\n")
    if server_id == "prod-db-01":
        return "CPU Load: 98% (CRITICAL)"
    return "CPU Load: 45% (NORMAL)"


@tool
def check_application_logs(server_id: str) -> str:
    """Retrieves the last 10 error logs from the server."""
    print(f"  [TOOL] Reading logs for {server_id}...\n")
    if server_id == "prod-db-01":
        return "Log: 'Connection pool exhausted', 'Timeout waiting for connection'"
    return "No critical errors found."


tools = [check_cpu_usage, check_application_logs]
```

---

## Step 5: Setup LLM

Create the LLM client. No need to bind tools manually - `create_agent` handles this automatically.

```python
# --- 5. Setup LLM ---
llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=API_URL)
```

---

## Step 6: Create the Agent

This is where the magic happens. One function call creates the entire agent.

```python
# --- 6. Create the Agent ---
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a senior site reliability engineer. Diagnose issues efficiently.",
)
```

**What `create_agent` does automatically:**

- Binds tools to the LLM
- Creates the state management
- Sets up the LLM node
- Sets up the tool execution node
- Configures the routing logic
- Compiles the graph

---

## Step 7: Run the Agent

```python
# --- 7. Run the Agent ---
def run_agent(user_input: str):
    print(f"\n[AGENT] Started with query: {user_input}\n")

    result = agent.invoke({"messages": [("user", user_input)]})

    # Print final response
    print(f"[AGENT] Final response:\n{result['messages'][-1].content}\n")

    # Debug the conversation history
    # print("[DEBUG] Conversation history:")
    # for msg in result["messages"]:
    #     msg.pretty_print()
    # print()
```

**Key Points:**

- Input is just `{"messages": [("user", user_input)]}`
- `agent.invoke()` runs the entire loop until completion
- Result contains all messages from the conversation

---

## Step 8: Run It

```python
# --- 8. Run It ---
if __name__ == "__main__":
    if not API_KEY:
        print("Error: API_KEY not set.")
    else:
        run_agent("Investigate performance issues on prod-db-01")
```

Run it:

```bash
uv run 02-infra-doctor.py
```

---
