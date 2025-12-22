# 02 - Building an AI Agent with LangGraph: Infra Doctor

This guide walks you through building the same **Site Reliability Engineer (SRE)** agent, but using **LangChain** and **LangGraph** instead of raw API calls.

> **Prerequisites:** Complete [00-environment-setup.md](./00-environment-setup.md) first.

Install dependencies:

```bash
uv add langchain-openai langgraph python-dotenv
```

---

## Step 1: Import Libraries

We import LangChain for the LLM and tools, and LangGraph for building the workflow graph.

```python
import json
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
```

**Key imports:**

- `ChatOpenAI` - LLM client (works with OpenAI-compatible APIs)
- `@tool` - Decorator to define tools
- `StateGraph`, `START`, `END` - LangGraph components for building the workflow
- `ToolNode`, `tools_condition` - Prebuilt components that handle tool execution and routing

---

## Step 2: Load Environment Variables

Load your API key from the `.env` file.

```python
load_dotenv()
```

---

## Step 3: Setup API Configuration

```python
API_KEY = os.environ.get("API_KEY", "")
API_URL = "https://truefoundry.staging.sunbit.in/api/llm"
MODEL = "vertex-staging/gemini-2-5-flash"
```

---

## Step 4: Define the Tools

With LangChain, we use the `@tool` decorator. It automatically generates the JSON schema from the function signature and docstring.

```python
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

**Key Points:**

- The `@tool` decorator replaces the manual `tools_schema` JSON
- The docstring becomes the tool's `description`
- Type hints become the parameter schema

---

## Step 5: Setup LLM with Tools

Create the LLM client and bind the tools to it.

```python
llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=API_URL)
llm_with_tools = llm.bind_tools(tools)
```

**Key Points:**

- `ChatOpenAI` works with any OpenAI-compatible endpoint via `base_url`
- `bind_tools()` tells the LLM what tools are available

---

## Step 6: Define the State

The state holds all data passed between nodes. We use `add_messages` to automatically handle appending messages.

```python
class State(TypedDict):
    # 'add_messages' automatically appends new messages to history
    messages: Annotated[list, add_messages]
```

**Key Points:**

- `Annotated[list, add_messages]` tells LangGraph how to merge message updates
- No need to manually do `state["messages"] + [new_message]`

---

## Step 7: Define Nodes

We only need to define one custom node (the LLM). The tool execution uses a prebuilt `ToolNode`.

```python
def llm_node(state):
    """Call the LLM."""
    print("[LLM] Calling LLM...\n")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Prebuilt node that executes tools automatically
tool_node = ToolNode(tools)
```

**Key Points:**

- `llm_node` just returns the new message (not the full list, thanks to `add_messages`)
- `ToolNode(tools)` creates a node that automatically:
  - Reads tool calls from the last message
  - Executes the matching tools
  - Returns the results as `ToolMessage` objects

---

## Step 8: Build the Graph

Now we connect everything using the prebuilt `tools_condition` for routing.

```python
#
#   START ──▶ llm ──▶ tools_condition? ──▶ END
#              ▲            │
#              │           tools
#              │            │
#              └────────────┘
#
graph = StateGraph(State)

graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)  # Prebuilt routing logic
graph.add_edge("tools", "llm")

agent = graph.compile()
```

**Key Points:**

- `add_node()` - Registers a node function
- `add_edge()` - Creates a fixed path between nodes
- `tools_condition` - Prebuilt function that routes to `"tools"` if tool calls exist, otherwise to `END`
- `compile()` - Finalizes the graph into a runnable agent

---

## Step 9: Run the Agent

```python
def run_agent(user_input: str):
    print(f"\n[AGENT] Started with query: {user_input}\n")

    initial_state = {
        "messages": [
            SystemMessage(content="You are a senior site reliability engineer. Diagnose issues efficiently."),
            HumanMessage(content=user_input),
        ]
    }

    final_state = agent.invoke(initial_state)

    # Print final response
    print(f"[AGENT] Final response:\n{final_state['messages'][-1].content}\n")

    # Print conversation history
    print("[AGENT] Conversation history:")
    history = []
    for m in final_state["messages"]:
        entry = {"role": m.type, "content": m.content}
        if hasattr(m, "tool_calls") and m.tool_calls:
            entry["tool_calls"] = m.tool_calls
        history.append(entry)
    print(json.dumps(history, indent=2))
    print()
```

**Key Points:**

- We use LangChain message types (`SystemMessage`, `HumanMessage`)
- `agent.invoke()` runs the entire graph until it reaches `END`
- The final state contains all messages from the conversation

---

## Step 10: Run It

```python
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

## How It Works (Flow Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│        "Investigate performance issues on prod-db-01"       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        ┌───────────┐
                        │   START   │
                        └───────────┘
                              │
                              ▼
                        ┌───────────┐
                        │    llm    │  ◀──────────────┐
                        └───────────┘                 │
                              │                       │
                              ▼                       │
                    ┌─────────────────┐               │
                    │ tools_condition │               │
                    └─────────────────┘               │
                     /              \                 │
              has tools?         no tools             │
                   /                  \               │
                  ▼                    ▼              │
            ┌───────────┐        ┌───────────┐       │
            │   tools   │        │    END    │       │
            │ (ToolNode)│        └───────────┘       │
            └───────────┘                             │
                  │                                   │
                  └───────────────────────────────────┘
```

---

## Comparison: Manual vs LangGraph

| Aspect         | Manual (`01-infra-doctor.py`) | LangGraph (`02-infra-doctor.py`)   |
| -------------- | ----------------------------- | ---------------------------------- |
| LLM Call       | Raw HTTP with `requests`      | `ChatOpenAI` client                |
| Tool Schema    | Manual JSON definition        | `@tool` decorator (auto-generated) |
| Agent Loop     | `while True` loop             | Graph with nodes and edges         |
| State          | `messages` list               | `State` with `add_messages`        |
| Flow Control   | `if/else` statements          | `tools_condition` (prebuilt)       |
| Tool Execution | Manual lookup and call        | `ToolNode` (prebuilt)              |

---

## What the Prebuilt Components Do

### `ToolNode(tools)`

Replaces this manual code:

```python
def tools_node(state):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = {t.name: t for t in tools}[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    return {"messages": results}
```

### `tools_condition`

Replaces this manual code:

```python
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END
```

### `add_messages`

Replaces manual message appending:

```python
# Before: return {"messages": state["messages"] + [response]}
# After:  return {"messages": [response]}  # add_messages handles appending
```
