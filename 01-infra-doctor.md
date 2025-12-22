# 01 - Building an AI Agent: Infra Doctor (Manual)

This guide walks you through building a simple AI agent that acts as a **Site Reliability Engineer (SRE)**. The agent can diagnose infrastructure issues by calling tools (functions) to check CPU usage and application logs.

> **Prerequisites:** Complete [00-environment-setup.md](./00-environment-setup.md) first.

Install dependencies:

```bash
uv add requests python-dotenv
```

---

## Step 1: Import Libraries

We start by importing the necessary libraries. We'll use `json` for parsing, `os` for environment variables, `requests` for HTTP calls, and `dotenv` to load environment variables from a `.env` file.

```python
import json
import os
import requests
from dotenv import load_dotenv
```

---

## Step 2: Load Environment Variables

Load your API key from the `.env` file.

```python
load_dotenv()
```

---

## Step 3: Setup API Configuration

Configure the API endpoint, your API key, and the model you want to use. This setup uses a standard OpenAI-compatible chat completions endpoint.

```python
API_KEY = os.environ.get("API_KEY", "")
API_URL = "https://truefoundry.staging.sunbit.in/api/llm/chat/completions"
MODEL = "vertex-staging/gemini-2-5-flash"
```

---

## Step 4: Define the LLM Call Function

This function wraps the HTTP request to the LLM. It takes a payload (containing messages and tools), sends it to the API, and returns the JSON response.

```python
def call_llm(payload):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()
```

**Key Points:**

- We set the `Authorization` header with a Bearer token
- `response.raise_for_status()` will throw an exception if the request fails
- The response is parsed as JSON and returned

---

## Step 5: Define the Tools (Functions)

Tools are Python functions that the agent can call to interact with the real world. In this example, we define two tools:

1. **check_cpu_usage** - Checks CPU usage for a server
2. **check_application_logs** - Retrieves error logs from a server

```python
def check_cpu_usage(server_id: str):
    """Checks real-time CPU usage for a specific server."""
    print(f"  [TOOL] Checking CPU for {server_id}...\n")
    if server_id == "prod-db-01":
        return "CPU Load: 98% (CRITICAL)"
    return "CPU Load: 45% (NORMAL)"

def check_application_logs(server_id: str):
    """Retrieves the last 10 error logs from the server."""
    print(f"  [TOOL] Reading logs for {server_id}...\n")
    if server_id == "prod-db-01":
        return "Log: 'Connection pool exhausted', 'Timeout waiting for connection'"
    return "No critical errors found."
```

**Key Points:**

- Each function takes typed parameters
- Functions return string results that the LLM can understand
- In a real scenario, these would call actual monitoring APIs

---

## Step 6: Create the Available Functions Dictionary

This dictionary maps function names (as strings) to the actual Python function objects. The agent uses this to execute the correct function when the LLM requests a tool call.

```python
available_functions = {
    "check_cpu_usage": check_cpu_usage,
    "check_application_logs": check_application_logs,
}
```

---

## Step 7: Define the Tools Schema

The tools schema tells the LLM what tools are available and how to use them. This follows the OpenAI function calling format.

```python
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "check_cpu_usage",
            "description": "Checks real-time CPU usage for a specific server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "The ID of the server."}
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
                    "server_id": {"type": "string", "description": "The ID of the server."}
                },
                "required": ["server_id"],
            },
        },
    },
]
```

**Key Points:**

- Each tool has a `name`, `description`, and `parameters` schema
- The `description` helps the LLM understand when to use each tool
- `parameters` follows JSON Schema format to define expected inputs

---

## Step 8: Initialize the Agent State

Now we start building the agent loop. First, create the `run_agent` function and initialize the messages list. This list acts as the agent's memory/context.

```python
def run_agent(user_input: str):
    # Initialize the messages list with the system prompt and user input
    messages = [
        {
            "role": "system",
            "content": "You are a senior site reliability engineer. Diagnose issues efficiently."
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    print(f"\n[AGENT] Started with query: {user_input}\n")
```

**Key Points:**

- The **system prompt** defines the agent's persona and behavior
- The **user message** contains the task or question
- This messages list will grow as the agent thinks and uses tools

---

## Step 9: Start the Agent Loop

Add the main loop structure. The agent will keep looping until it has a final response (no more tool calls).

```python
    # Continue from Step 8...

    while True:
        # Construct the payload for the LLM call
        payload = {
            "model": MODEL,
            "messages": messages,
            "tools": tools_schema
        }

        # Call the LLM
        print("[LLM] Calling LLM...\n")
        try:
            response = call_llm(payload)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            break
```

**Key Points:**

- Each iteration sends the full conversation history to the LLM
- The `tools` parameter tells the LLM what functions it can call
- Error handling prevents crashes if the API fails

---

## Step 10: Parse the LLM Response

Extract the response content and check if the LLM wants to call any tools.

```python
        # Continue inside the while loop...

        # Parse the LLM Response
        response_message = response["choices"][0]["message"]
        content = response_message.get("content")
        tool_calls = response_message.get("tool_calls")
```

**Key Points:**

- `content` contains the LLM's text response (if any)
- `tool_calls` is a list of tools the LLM wants to execute (if any)
- The LLM decides whether to respond with text or call tools

---

## Step 11: Handle Tool Calls

If the LLM requested tool calls, execute each tool and add the results back to the conversation.

```python
        # Continue inside the while loop...

        if tool_calls:
            # First, append the assistant message that contains the tool calls
            messages.append(response_message)

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                print(f"[CALLING TOOL] {tool_name}\n")

                tool_args = json.loads(tool_call["function"]["arguments"])

                # Execute the tool
                tool_result = available_functions[tool_name](**tool_args)

                print(f"  [RESULT] {tool_result}\n")

                # Add the tool result to the conversation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "content": tool_result,
                    }
                )
```

**Key Points:**

- We first append the assistant's message (containing `tool_calls`) to preserve conversation structure
- `json.loads()` parses the arguments from the LLM
- `**tool_args` unpacks the arguments into the function call
- Tool results are added to messages so the LLM can see them in the next iteration

---

## Step 12: Handle Final Response

If no tool calls were requested, the LLM has finished reasoning. Save the response and exit the loop.

```python
        # Continue inside the while loop...

        else:
            # No tool calls - the LLM has a final response
            print(f"[AGENT] Final response:\n{content}\n")
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
            break  # Exit the loop and return the final response

    # Pretty print the full conversation
    print("[AGENT] Conversation history:")
    print(json.dumps(messages, indent=2))
    print()
```

**Key Points:**

- The `else` block runs when `tool_calls` is empty/None
- `break` exits the while loop, ending the agent's reasoning
- The final response is stored in the messages list
- After the loop, we pretty-print the full conversation history for debugging

---

## Step 13: Run the Agent

Finally, add the entry point to run the agent. This checks that the API key is set and then starts the agent with a sample query.

```python
if __name__ == "__main__":
    if not API_KEY:
        print("Error: API_KEY not set.")
    else:
        run_agent("Investigate performance issues on prod-db-01")
```

Run it:

```bash
uv run 01-infra-doctor.py
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
