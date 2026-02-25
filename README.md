# Audit Microservice — Loriaa

AI-powered real-estate audit microservice for **LiveNjoy Inc**.  
Upload rent roll, rent projection, and concessions documents and let a **LangGraph ReAct agent** (backed by GPT-4o) automatically identify anomalies, discrepancies, and compliance issues.

---

## Features

| Feature | Details |
|---------|---------|
| **File uploads** | CSV, Excel (`.xlsx`/`.xls`), PDF, Word (`.docx`) |
| **Document types** | Rent Roll · Rent Projections · Concessions |
| **AI engine** | LangGraph ReAct agent + OpenAI GPT-4o |
| **Anomaly detection** | Missing fields · outliers · duplicate units · unapproved concessions · projection variance · expired leases |
| **Results UI** | Full report · structured anomaly list · raw agent output |

---

## Project Structure

```
audit-microservice-loriaa/
├── app.py                   # Streamlit UI entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── agents/
│   ├── __init__.py
│   └── audit_agent.py       # LangGraph ReAct audit agent + tools
├── parsers/
│   ├── __init__.py
│   └── document_parser.py   # Multi-format document parser
├── utils/
│   ├── __init__.py
│   └── data_processor.py    # Data normalisation & summarisation
└── tests/
    ├── test_document_parser.py
    └── test_data_processor.py
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd audit-microservice-loriaa
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Alternatively, enter the key directly in the sidebar when running the app.

### 3. Run the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. **Upload** one or more files into the **Rent Roll**, **Rent Projections**, or **Concessions** panels.
2. Enter your **OpenAI API key** in the sidebar (or set `OPENAI_API_KEY` in `.env`).
3. Click **🔍 Run AI Audit**.
4. Review results across three tabs: *Full Report*, *Anomaly List*, and *Raw Agent Output*.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | — | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o` | Model to use |
| `OPENAI_MAX_TOKENS` | No | `4096` | Max tokens per LLM response |
