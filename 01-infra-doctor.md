# 01 - Building an AI Agent: Infra Doctor (Manual)

This guide walks you through building a simple AI agent that acts as a **Site Reliability Engineer (SRE)**. The agent can diagnose infrastructure issues by calling tools (functions) to check CPU usage and application logs.

## How It Works (Flow Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│        "Investigate performance issues on prod-db-01"       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     AGENT LOOP STARTS                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CALL LLM WITH TOOLS                       │
│    (System prompt + User message + Tools schema)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Tool Calls?    │
                    └─────────────────┘
                     /              \
                   YES               NO
                   /                  \
                  ▼                    ▼
    ┌────────────────────┐    ┌────────────────────┐
    │  Execute Tool(s)   │    │  Return Response   │
    │  Add to Messages   │    │  EXIT LOOP         │
    └────────────────────┘    └────────────────────┘
                  │
                  └──────────► (Loop back to LLM call)
```

---

> **Prerequisites:** Complete [00-environment-setup.md](./00-environment-setup.md) first.

Install dependencies:

```bash
uv add requests python-dotenv
```

---

## Step 0: Create the Python File

Create a new file called `01-infra-doctor.py`:

```bash
touch 01-infra-doctor.py
```

Open the file in Cursor and follow the steps below.

---

## Step 1: Import Libraries

```python
# --- 1. Import Libraries ---
import json
import os
import requests
from dotenv import load_dotenv
```

---

## Step 2: Load Environment Variables

```python
# --- 2. Load Environment Variables ---
# This is a simple way to load environment variables from a .env file.
load_dotenv()
```

---

## Step 3: Setup API Configuration

```python
# --- 3. Setup API Configuration ---
# We are using the raw REST API, so we need the endpoint and API key
API_KEY = os.environ.get("API_KEY", "")  # Ensure this is set in your environment
API_URL = "https://truefoundry.staging.sunbit.in/api/llm/chat/completions"
MODEL = "vertex-staging/gemini-2-5-flash"
```

---

## Step 4: Define the LLM Call Function

```python
# --- 4. Define the LLM Call Function ---
# This function formats the request and makes the API call.
def call_llm(payload):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()
```

---

## Step 5: Define the Tools

```python
# --- 5. Define the Tools ---
# These are the functions that the agent can use to perform tasks.
def check_cpu_usage(server_id: str):
    print(f"[TOOL] Checking CPU for {server_id}...\n")
    if server_id == "prod-db-01":
        return "CPU Load: 98% (CRITICAL)"
    return "CPU Load: 45% (NORMAL)"


def check_application_logs(server_id: str):
    print(f"[TOOL] Reading logs for {server_id}...\n")
    if server_id == "prod-db-01":
        return "Log: 'Connection pool exhausted', 'Timeout waiting for connection'"
    return "No critical errors found."
```

---

## Step 6: Define the Available Functions

```python
# --- 6. Define the Available Functions ---
# This is the dictionary of functions that the agent can use.
available_functions = {
    "check_cpu_usage": check_cpu_usage,
    "check_application_logs": check_application_logs,
}
```

---

## Step 7: Define the Tools Schema

```python
# --- 7. Define the Tools Schema ---
# This is the schema for the tools that the agent can use.
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "check_cpu_usage",
            "description": "Checks real-time CPU usage for a specific server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "The ID of the server.",
                    }
                },
                "required": ["server_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_application_logs",
            "description": "Retrieves the last 10 error logs from the server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "The ID of the server.",
                    }
                },
                "required": ["server_id"],
            },
        },
    },
]
```

---

## Step 8: Define the System Message

```python
# --- 8. Define the System Message ---
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a senior site reliability engineer. Diagnose issues efficiently.",
}
```

---

## Step 9: The Agent

```python
# --- 9. The Agent ---
# This is the control flow of the agent.
def run_agent(user_input: str):
    messages = [SYSTEM_MESSAGE, {"role": "user", "content": user_input}]

    print(f"\n[AGENT] Started with query: {user_input}\n")

    while True:
        print("[LLM] Calling LLM...\n")
        response = call_llm(
            {"model": MODEL, "messages": messages, "tools": tools_schema}
        )

        last_message = response["choices"][0]["message"]

        messages.append(last_message)

        tool_calls = last_message.get("tool_calls")
        content = last_message.get("content")

        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_result = available_functions[tool_name](**tool_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "content": tool_result,
                    }
                )
        else:
            print(f"[AGENT] Final response:\n{content}\n")
            break

    # debug the conversation history
    # print("[DEBUG] Conversation history:")
    # print(json.dumps(messages, indent=2))
```

---

## Step 10: Run It

```python
# --- 10. Run It ---
if __name__ == "__main__":
    # Ensure you have set the API key in your environment before running
    if not API_KEY:
        print("Error: API_KEY not set.")
    else:
        run_agent("Investigate performance issues on prod-db-01")
```

Run it:

```bash
uv run 01-infra-doctor.py
```
