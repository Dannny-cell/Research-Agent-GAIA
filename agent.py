"""Robust LangGraph agent for GAIA-style questions."""

from __future__ import annotations

import os
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import ArxivLoader, WikipediaLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_tavily import TavilySearch
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from supabase.client import Client, create_client

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_CANDIDATES = [
    PROJECT_DIR / "System_prompt.txt",
    PROJECT_DIR / "system_prompt.txt",
]
DEFAULT_PROVIDER = os.getenv("GAIA_LLM_PROVIDER", "groq").strip().lower()
DEFAULT_RECURSION_LIMIT = int(os.getenv("GAIA_RECURSION_LIMIT", "20"))
MAX_CONTEXT_MESSAGES = int(os.getenv("GAIA_MAX_CONTEXT_MESSAGES", "10"))
REFERENCE_DOC_LIMIT = int(os.getenv("GAIA_REFERENCE_DOC_LIMIT", "1"))
REFERENCE_DOC_CHARS = int(os.getenv("GAIA_REFERENCE_DOC_CHARS", "900"))
USE_SUPABASE_REFERENCE = os.getenv("GAIA_USE_SUPABASE_REFERENCE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
EMBEDDING_MODEL_NAME = os.getenv(
    "GAIA_EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2",
)
WEB_RESULT_CONTENT_CHARS = int(os.getenv("GAIA_WEB_RESULT_CONTENT_CHARS", "450"))

PROMPT_APPENDIX = """
You are solving GAIA-style questions.

Rules:
1. Use tools when the question depends on external facts, calculations, or current webpages.
2. Prefer verified evidence over memory.
3. If a retrieved example is included, use it only as a hint.
4. The final line must be exactly: FINAL ANSWER: <answer>
5. Do not add any text after the final answer line.
""".strip()

DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant tasked with answering questions using a set of tools.

Your final answer must strictly follow this format:
FINAL ANSWER: [ANSWER]

Only write the answer in that exact format. Do not explain anything. Do not include any other text.

If you are provided with a similar question and its final answer, and the current question is exactly the same,
then simply return the same final answer without using any tools.

Only use tools if the current question is different from the similar one.
""".strip()

TOOLCALL_REPAIR_PROMPT = (
    "Tool-use repair instruction: use only the model's native structured tool calls. "
    "Do not write manual function syntax, XML-like tags, JSON snippets, or <function=...> blocks."
)

DIRECT_FALLBACK_PROMPT = """
Answer the user's question directly without using any tools.
Use only the content in the prompt and ordinary reasoning.
If the question depends on an unavailable image, audio file, video details, or attached document, return a very short answer instead of an explanation.
The final line must be exactly: FINAL ANSWER: <answer>
Do not add any text after that line.
""".strip()


def _coerce_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numeric inputs.")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def _format_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _truncate(text: str, limit: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _compact_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    if len(messages) <= MAX_CONTEXT_MESSAGES + 1:
        return messages
    if messages and isinstance(messages[0], SystemMessage):
        return [messages[0], *messages[-MAX_CONTEXT_MESSAGES:]]
    return messages[-MAX_CONTEXT_MESSAGES:]


def _format_loaded_documents(docs: Iterable[Any], label: str, char_limit: int = 900) -> str:
    formatted_docs: list[str] = []
    for index, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("entry_id") or label
        title = metadata.get("title") or metadata.get("page") or f"{label}-{index}"
        content = _truncate(getattr(doc, "page_content", ""), char_limit)
        formatted_docs.append(f"[{label} #{index}] title={title} source={source}\n{content}")
    return "\n\n---\n\n".join(formatted_docs) if formatted_docs else f"{label}: no results"


def _format_tavily_results(raw_results: Any) -> str:
    if isinstance(raw_results, dict):
        results = raw_results.get("results", [])
    else:
        results = raw_results or []

    formatted: list[str] = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            formatted.append(f"[web #{index}] {_truncate(str(item), WEB_RESULT_CONTENT_CHARS)}")
            continue
        title = item.get("title", f"result-{index}")
        url = item.get("url", "unknown")
        content = _truncate(item.get("content") or item.get("snippet") or "", WEB_RESULT_CONTENT_CHARS)
        formatted.append(f"[web #{index}] title={title} url={url}\n{content}")

    return "\n\n---\n\n".join(formatted) if formatted else "web_search: no results"


def _load_system_prompt() -> str:
    custom_path = os.getenv("GAIA_SYSTEM_PROMPT_PATH", "").strip()
    if custom_path:
        prompt_path = Path(custom_path)
        if prompt_path.exists():
            base_prompt = prompt_path.read_text(encoding="utf-8").strip()
            return f"{base_prompt}\n\n{PROMPT_APPENDIX}"

    for prompt_path in SYSTEM_PROMPT_CANDIDATES:
        if prompt_path.exists():
            base_prompt = prompt_path.read_text(encoding="utf-8").strip()
            return f"{base_prompt}\n\n{PROMPT_APPENDIX}"

    base_prompt = DEFAULT_SYSTEM_PROMPT
    return f"{base_prompt}\n\n{PROMPT_APPENDIX}"


def _get_system_message() -> SystemMessage:
    return SystemMessage(content=_load_system_prompt())


def _latest_user_question(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def _extract_balanced_json_like(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _manual_tool_message_from_text(text: str) -> Optional[AIMessage]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    normalized_text = cleaned.replace('""', '"')

    tool_names = "|".join(tool.name for tool in TOOLS)
    match = re.search(rf"(?:<function=)?(?P<name>{tool_names})", normalized_text)
    if not match:
        return None

    tool_name = match.group("name")
    args: dict[str, Any] = {}

    json_like = _extract_balanced_json_like(normalized_text[match.start() :])
    if json_like:
        normalized = json_like
        normalized = re.sub(r",\s*}", "}", normalized)
        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, dict):
                args = parsed
        except json.JSONDecodeError:
            pass

    if not args:
        query_match = re.search(r'query["\']?\s*[:=]\s*["\'](.+?)["\']', normalized_text, flags=re.IGNORECASE)
        if query_match:
            args = {"query": query_match.group(1)}

    if tool_name in {"web_search", "wiki_search", "arxiv_search"} and "query" not in args:
        return None

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": args,
                "id": f"manual_{tool_name}_call",
            }
        ],
    )


def _is_self_contained_question(question: str) -> bool:
    lowered = (question or "").lower()
    external_markers = [
        "http://",
        "https://",
        "wikipedia",
        "youtube",
        "video",
        "image",
        "audio",
        ".mp3",
        ".wav",
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".xlsx",
        ".csv",
        "attached",
        "attachment",
        "article",
        "paper",
        "published",
        "as of ",
        "latest ",
        "find this paper",
    ]
    return not any(marker in lowered for marker in external_markers)


@lru_cache(maxsize=1)
def _get_vector_store() -> Optional[SupabaseVectorStore]:
    if not USE_SUPABASE_REFERENCE:
        return None

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        supabase: Client = create_client(supabase_url, supabase_key)
        return SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name=os.getenv("SUPABASE_TABLE_NAME", "documents"),
            query_name=os.getenv("SUPABASE_QUERY_NAME", "match_documents_langchain"),
        )
    except Exception:
        return None


def _reference_message(question: str) -> Optional[HumanMessage]:
    vector_store = _get_vector_store()
    if vector_store is None or not question.strip():
        return None

    try:
        similar_docs = vector_store.similarity_search(question, k=REFERENCE_DOC_LIMIT)
    except Exception:
        return None

    if not similar_docs:
        return None

    reference_block = _format_loaded_documents(
        similar_docs,
        label="reference",
        char_limit=REFERENCE_DOC_CHARS,
    )
    return HumanMessage(
        content=(
            "Reference example from the vector store. "
            "Use it only if it helps with the current question.\n\n"
            f"{reference_block}"
        )
    )


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers."""
    try:
        return f"multiply: {_format_number(_coerce_number(a) * _coerce_number(b))}"
    except Exception as exc:
        return f"multiply_error: {exc}"


@tool
def add(a: float, b: float) -> str:
    """Add two numbers."""
    try:
        return f"add: {_format_number(_coerce_number(a) + _coerce_number(b))}"
    except Exception as exc:
        return f"add_error: {exc}"


@tool
def subtract(a: float, b: float) -> str:
    """Subtract two numbers."""
    try:
        return f"subtract: {_format_number(_coerce_number(a) - _coerce_number(b))}"
    except Exception as exc:
        return f"subtract_error: {exc}"


@tool
def divide(a: float, b: float) -> str:
    """Divide two numbers."""
    try:
        numerator = _coerce_number(a)
        denominator = _coerce_number(b)
        if denominator == 0:
            return "divide_error: Cannot divide by zero."
        return f"divide: {_format_number(numerator / denominator)}"
    except Exception as exc:
        return f"divide_error: {exc}"


@tool
def modulus(a: float, b: float) -> str:
    """Return the modulus of two numbers."""
    try:
        left = _coerce_number(a)
        right = _coerce_number(b)
        if right == 0:
            return "modulus_error: Cannot take modulus by zero."
        return f"modulus: {_format_number(left % right)}"
    except Exception as exc:
        return f"modulus_error: {exc}"


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia and return a compact summary of up to 2 pages."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return "wiki_search_error: query must not be empty"

    try:
        docs = WikipediaLoader(query=cleaned_query, load_max_docs=2).load()
        return _format_loaded_documents(docs, "wikipedia", char_limit=800)
    except Exception as exc:
        return f"wiki_search_error: {exc}"


@tool
def web_search(query: str) -> str:
    """Search the web with one string query and return compact results."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return "web_search_error: query must not be empty"

    try:
        tavily = TavilySearch(max_results=3)
        try:
            raw_results = tavily.invoke({"query": cleaned_query})
        except Exception:
            raw_results = tavily.invoke(cleaned_query)
        return _format_tavily_results(raw_results)
    except Exception as exc:
        return f"web_search_error: {exc}"


@tool
def arxiv_search(query: str) -> str:
    """Search arXiv and return up to 3 compact abstracts."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return "arxiv_search_error: query must not be empty"

    try:
        docs = ArxivLoader(query=cleaned_query, load_max_docs=3).load()
        return _format_loaded_documents(docs, "arxiv", char_limit=700)
    except Exception as exc:
        return f"arxiv_search_error: {exc}"


TOOLS = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    web_search,
    arxiv_search,
]


def _get_llm(provider: str):
    provider_name = (provider or DEFAULT_PROVIDER).strip().lower()
    if provider_name == "google":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=0,
        )
    if provider_name == "groq":
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            max_tokens=1024,
        )
    if provider_name == "huggingface":
        endpoint = HuggingFaceEndpoint(
            repo_id=os.getenv("HUGGINGFACE_REPO_ID", "Qwen/Qwen2.5-Coder-32B-Instruct"),
            task="text-generation",
            temperature=0,
            max_new_tokens=int(os.getenv("HUGGINGFACE_MAX_NEW_TOKENS", "1024")),
        )
        return ChatHuggingFace(llm=endpoint)
    raise ValueError("Invalid provider. Choose 'google', 'groq', or 'huggingface'.")


@lru_cache(maxsize=6)
def _get_bound_llm(provider: str):
    return _get_llm(provider).bind_tools(TOOLS)


@lru_cache(maxsize=6)
def _get_plain_llm(provider: str):
    return _get_llm(provider)


def _is_toolcall_error(error_text: str) -> bool:
    lowered = (error_text or "").lower()
    return "tool_use_failed" in lowered or "failed to call a function" in lowered


def _with_toolcall_repair(messages: list[BaseMessage]) -> list[BaseMessage]:
    repair_message = SystemMessage(content=TOOLCALL_REPAIR_PROMPT)
    if any(
        isinstance(message, SystemMessage) and TOOLCALL_REPAIR_PROMPT in str(message.content)
        for message in messages
    ):
        return messages
    if messages and isinstance(messages[0], SystemMessage):
        return [messages[0], repair_message, *messages[1:]]
    return [repair_message, *messages]


def _invoke_direct_answer(question: str, provider: str) -> str:
    prompt = (
        f"{DIRECT_FALLBACK_PROMPT}\n\n"
        f"Question:\n{question.strip()}"
    )
    response = _get_plain_llm(provider).invoke([HumanMessage(content=prompt)])
    if isinstance(response, AIMessage):
        return str(response.content).strip()
    return str(response).strip()


def _invoke_with_fallback(messages: list[BaseMessage], provider: str) -> AIMessage:
    primary = _get_bound_llm(provider)
    try:
        response = primary.invoke(messages)
    except Exception as exc:
        error_text = str(exc)
        manual_tool_message = _manual_tool_message_from_text(error_text)
        if manual_tool_message is not None:
            return manual_tool_message

        if not _is_toolcall_error(error_text):
            raise

        repaired_messages = _with_toolcall_repair(messages)
        fallback_provider = os.getenv("GAIA_TOOLCALL_FALLBACK_PROVIDER", "").strip().lower()

        try:
            response = primary.invoke(repaired_messages)
        except Exception:
            if provider.strip().lower() == "groq" and fallback_provider and fallback_provider != "groq":
                response = _get_bound_llm(fallback_provider).invoke(repaired_messages)
            else:
                direct_answer = _invoke_direct_answer(_latest_user_question(messages), provider)
                response = AIMessage(content=direct_answer)

    if not isinstance(response, AIMessage):
        return AIMessage(content=str(response))

    manual_tool_message = _manual_tool_message_from_text(str(response.content))
    if manual_tool_message is not None and not response.tool_calls:
        return manual_tool_message

    return response


def normalize_final_answer(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return "FINAL ANSWER: "

    match = None
    for match in re.finditer(r"FINAL ANSWER:\s*(.*)", cleaned, flags=re.IGNORECASE | re.DOTALL):
        pass
    if match:
        answer = match.group(1).strip()
    else:
        answer = " ".join(part for part in cleaned.splitlines() if part.strip()).strip()

    return f"FINAL ANSWER: {answer}"


def _extract_last_ai_message(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content).strip()
    raise RuntimeError("The agent finished without producing an AI message.")


@lru_cache(maxsize=3)
def build_graph(provider: str = DEFAULT_PROVIDER):
    """Build and compile the LangGraph workflow."""
    system_message = _get_system_message()

    def retriever(state: MessagesState) -> dict[str, list[Any]]:
        messages = [system_message, *state["messages"]]
        question = _latest_user_question(state["messages"])
        reference = _reference_message(question)
        if reference is not None:
            messages.append(reference)
        return {"messages": _compact_messages(messages)}

    def assistant(state: MessagesState) -> dict[str, list[AIMessage]]:
        question = _latest_user_question(state["messages"])
        if _is_self_contained_question(question):
            direct_answer = _invoke_direct_answer(question, provider)
            return {"messages": [AIMessage(content=direct_answer)]}

        response = _invoke_with_fallback(_compact_messages(state["messages"]), provider)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


def run_agent(
    question: str,
    provider: str = DEFAULT_PROVIDER,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> str:
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        return "FINAL ANSWER: No question provided"

    graph = build_graph(provider)
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=cleaned_question)]},
            config={"recursion_limit": recursion_limit},
        )
        final_text = _extract_last_ai_message(result)
        return normalize_final_answer(final_text)
    except GraphRecursionError:
        return normalize_final_answer(_invoke_direct_answer(cleaned_question, provider))


if __name__ == "__main__":
    demo_question = "What is 12 times 8?"
    print(run_agent(demo_question, provider=DEFAULT_PROVIDER))
