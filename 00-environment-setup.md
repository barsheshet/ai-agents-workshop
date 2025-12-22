# Environment Setup

This guide walks you through setting up your development environment for the AI Agents Workshop.

---

## 1. Install uv (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Create a New Project

Open a new folder in Cursor, then initialize a Python project:

```bash
uv init
```

> Note: If you have safe-chain installed, use `command uv init`

---

## 3. Verify Setup

Run the main.py file to verify everything works:

```bash
uv run main.py
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```bash
touch .env
```

Add your API key to the `.env` file:

```
API_KEY=your_api_key_here
```

Add `.env` to `.gitignore` to keep your secrets safe:

```bash
echo ".env" >> .gitignore
```

---

## Troubleshooting

### "API_KEY not set" error

Make sure your `.env` file exists and contains:

```
API_KEY=your_actual_api_key
```

### Import errors

Run `uv sync` to install all dependencies from `pyproject.toml`.

### "command not found: uv"

Restart your terminal after installing uv, or run:

```bash
source ~/.bashrc   # or ~/.zshrc for zsh
```

If you have `safe-chain` installed, use `command uv` to bypass the alias.
