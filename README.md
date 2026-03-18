# GAIA Research Agent 🤖

A high-performance, tool-augmented AI agent built using **LangGraph** and **LangChain** to solve complex, multi-step reasoning questions from the **GAIA benchmark**. This agent is designed to perform research, use external tools dynamically, and ensure outputs follow a strict format.

---

## 🌟 Project Overview

The **GAIA Research Agent** tackles "hidden" reasoning problems that require more than just internal model knowledge. It follows a structured protocol: analyze the question, determine if external tools (Search, arXiv, Math) are needed, execute the research, and synthesize a precise response.

**Core Goal:** Solve complex reasoning problems and return:  
`FINAL ANSWER: <answer>`

---

## 🏗️ Architecture & Workflow

The system is built as a stateful **LangGraph** workflow with the following nodes:

1.  **Retriever Node**: Injects the system prompt, compacts message history, and optionally retrieves similar examples from a **Supabase** vector store (RAG).
2.  **Assistant Node**: The "brain" of the agent. It uses an LLM to decide reasoning steps and whether to call tools or answer directly.
3.  **Tool Node**: Executes external tools such as:
    * **Math Operations**: Add, subtract, multiply, divide, modulus.
    * **Search**: Tavily Web Search, Wikipedia, and arXiv.

**The Flow:** `User Question` → `Retriever` → `Assistant` ↔ `Tools (if needed)` → `Assistant` → `FINAL ANSWER`

---

## 🚀 Key Features

* **Multi-LLM Support**: Configurable for **Groq** (LLaMA models), **Google Gemini**, and **HuggingFace** endpoints.
* **Tool-Call Repair**: Automatically fixes malformed tool calls using regex and fallback prompts.
* **Direct Fallback**: If tools fail or recursion limits are hit, the agent defaults to internal reasoning.
* **Context Management**: Limits message history to avoid token overflow.
* **RAG (Retrieval-Augmented Generation)**: Optional similarity search via Supabase to improve reasoning accuracy.

---

## 📂 File Structure

| File | Description |
| :--- | :--- |
| `agent.py` | Full LangGraph logic, tools, LLM setup, and fallback handling. |
| `app.py` | Gradio UI for running the evaluation suite and scoring. |
| `System_prompt.txt` | Custom system prompt defining the agent's persona. |
| `requirements.txt` | Python dependencies. |

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/GAIA-Research-Agent.git](https://github.com/your-username/GAIA-Research-Agent.git)
cd GAIA-Research-Agent
