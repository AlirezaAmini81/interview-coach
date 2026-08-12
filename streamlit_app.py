"""
Streamlit UI for the interview coach. Pure presentation layer - all the
routing/RAG/scoring logic still lives in src/orchestrator and src/agents,
untouched. Run with `streamlit run streamlit_app.py` from the project root.
"""
import os

import streamlit as st
from dotenv import load_dotenv

from src.agents.must_have_extractor_agent import extract_must_haves
from src.llm.mock_provider import MockProvider
from src.orchestrator.graph import build_graph
from src.orchestrator.session_state import SessionState
from src.rag.embeddings import OllamaEmbedder, TfidfEmbedder
from src.rag.index import build_indices

load_dotenv()

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
CV_PATH = os.path.join(os.path.dirname(__file__), "data", "cv_placeholder.txt")
JOB_POSTING_PATH = os.path.join(os.path.dirname(__file__), "data", "job_posting.txt")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), ".chroma_data")
PLACEHOLDER_MARKER = "[Replace this entire section"

st.set_page_config(page_title="Interview Coach", page_icon="🎤", layout="wide")


@st.cache_resource(show_spinner="Building RAG indices...")
def get_indices(embedder_choice: str):
    embedder_factory = OllamaEmbedder if embedder_choice == "ollama" else TfidfEmbedder
    return build_indices(embedder_factory, persist_path=CHROMA_PATH)


@st.cache_resource(show_spinner="Connecting to LLM provider...")
def get_llm(provider_choice: str, ollama_model: str, api_key: str | None):
    if provider_choice == "anthropic":
        from src.llm.langchain_provider import LangChainProvider

        return LangChainProvider(model=DEFAULT_ANTHROPIC_MODEL, provider="anthropic", api_key=api_key)
    if provider_choice == "ollama":
        from src.llm.langchain_provider import LangChainProvider

        return LangChainProvider(model=ollama_model, provider="ollama")
    return MockProvider()


@st.cache_data(show_spinner="Reading job posting for must-haves...")
def get_must_haves(job_posting_text: str, _llm) -> list[str]:
    """Underscore-prefixed _llm is excluded from Streamlit's cache key
    hashing (it's not hashable/serializable) - cache is keyed on the
    posting text alone, which is what actually determines the result."""
    return extract_must_haves(job_posting_text, _llm)


def _read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


def default_cv_text() -> str:
    return _read_file(CV_PATH)


def default_job_posting_text() -> str:
    return _read_file(JOB_POSTING_PATH)


with st.sidebar:
    st.header("Setup")
    provider_choice = st.selectbox("LLM provider", ["mock", "ollama", "anthropic"])

    ollama_model = "llama3.2:1b"
    api_key = None
    if provider_choice == "ollama":
        ollama_model = st.text_input("Ollama model", ollama_model)
    elif provider_choice == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY") or st.text_input("Anthropic API key", type="password")

    embedder_choice = st.selectbox("Embedder", ["tfidf", "ollama"])

    job_posting_text = st.text_area(
        "Job posting",
        value=st.session_state.get("job_posting_text", default_job_posting_text()),
        height=160,
    )
    st.session_state["job_posting_text"] = job_posting_text
    st.caption(
        "Must-haves are extracted from this text (LLM call, cached per posting) "
        "instead of a fixed topic list - edit it to practice for a different role."
    )

    cv_text = st.text_area(
        "Your background (CV)",
        value=st.session_state.get("cv_text", default_cv_text()),
        height=220,
    )
    st.session_state["cv_text"] = cv_text
    if PLACEHOLDER_MARKER in cv_text:
        st.warning("Still the placeholder CV - questions and scoring won't be calibrated to your real background.")

    if st.button("Start / Restart session", type="primary"):
        keep_cv, keep_posting = cv_text, job_posting_text
        st.session_state.clear()
        st.session_state["cv_text"] = keep_cv
        st.session_state["job_posting_text"] = keep_posting
        st.rerun()

if provider_choice == "anthropic" and not api_key:
    st.warning("Enter your Anthropic API key in the sidebar to continue.")
    st.stop()

try:
    llm = get_llm(provider_choice, ollama_model, api_key)
    project_store, project_embedder, docs_store, docs_embedder = get_indices(embedder_choice)
except ConnectionError as exc:
    st.error(f"Could not reach Ollama: {exc}")
    st.stop()

graph = build_graph(llm, project_store, project_embedder, docs_store, docs_embedder)

if "session_state" not in st.session_state:
    must_haves = get_must_haves(job_posting_text, llm)
    st.session_state.session_state = SessionState(must_haves=must_haves)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_complete" not in st.session_state:
    st.session_state.session_complete = False

state: SessionState = st.session_state.session_state


def run_turn(user_message: str) -> dict:
    return graph.invoke(
        {
            "user_message": user_message,
            "cv_text": cv_text,
            "session_state": state,
            "response": "",
            "session_complete": False,
            "_mode": "",
        }
    )


with st.sidebar:
    st.header("Progress")
    st.progress(state.coverage_fraction(), text=f"{len(state.covered_topics)}/{len(state.must_haves)} topics covered")
    col1, col2 = st.columns(2)
    col1.metric("Avg score", f"{state.average_score():.1f}/5" if state.scored_answers else "-")
    col2.metric("Turns", state.turn_count)
    with st.expander("Must-haves", expanded=False):
        for topic in state.must_haves:
            icon = "✅" if topic in state.covered_topics else "⬜"
            st.write(f"{icon} {topic}")

st.title("🎤 Interview Coach")
st.caption(
    "Answer questions as they come, or ask something like "
    "'explain LangGraph conditional edges' or 'quiz me on RAG'."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    with st.spinner("Preparing your first question..."):
        result = run_turn("ready")
    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    state.record_turn("assistant", result["response"])
    if result.get("session_complete"):
        st.session_state.session_complete = True
    st.rerun()

if st.session_state.session_complete:
    st.info("Session complete. Use “Start / Restart session” in the sidebar to practice again.")
else:
    user_message = st.chat_input("Your answer or question...")
    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.spinner("Thinking..."):
            result = run_turn(user_message)
        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
        state.record_turn("user", user_message)
        state.record_turn("assistant", result["response"])
        if result.get("session_complete"):
            st.session_state.session_complete = True
        st.rerun()
