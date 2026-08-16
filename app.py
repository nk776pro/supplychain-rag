import streamlit as st
from pathlib import Path

from ingest import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from rag import ask_question


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "supply_chain"

st.set_page_config(
    page_title="Meridian Supply Chain AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# UI STYLING
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN PAGE
       ===================================================== */

    .stApp {
        background-color: #f5f7fa;
        color: #1e293b;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       MAIN HEADINGS
       ===================================================== */

    h1 {
        color: #0f172a !important;
        font-weight: 750 !important;
    }

    h2 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    .main p {
        color: #334155;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #d1d5db !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #374151;
    }


    /* =====================================================
       SIDEBAR INFO BOX
       ===================================================== */

    section[data-testid="stSidebar"]
    [data-testid="stAlert"] {
        background-color: #1f2937;
        border: 1px solid #374151;
    }

    section[data-testid="stSidebar"]
    [data-testid="stAlert"] p {
        color: #e5e7eb !important;
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    [data-testid="stMetric"] {
        background-color: #ffffff;

        border: 1px solid #e2e8f0;

        border-radius: 12px;

        padding: 14px;

        box-shadow:
            0 2px 7px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.75rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background-color: #ffffff;

        border: 1px solid #dbe3ee;

        border-radius: 12px;

        padding: 0.6rem;

        box-shadow:
            0 2px 7px rgba(15, 23, 42, 0.03);
    }

    [data-testid="stFileUploader"] label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
    }

    [data-testid="stFileUploader"] small {
        color: #64748b !important;
    }


    /* =====================================================
       TEXT AREA
       ===================================================== */

    [data-testid="stTextArea"] label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    [data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;

        color: #1e293b !important;

        border: 1px solid #cbd5e1 !important;

        border-radius: 10px !important;

        font-size: 0.95rem !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #94a3b8 !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        width: 100%;

        min-height: 44px;

        border-radius: 9px;

        font-weight: 650;

        background-color: #2563eb;

        color: #ffffff;

        border: 1px solid #2563eb;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;

        border-color: #1d4ed8;

        color: #ffffff;
    }


    /* =====================================================
       ALERTS
       ===================================================== */

    [data-testid="stAlert"] p {
        color: #1e293b !important;
    }


    /* =====================================================
       EXPANDER
       ===================================================== */

    [data-testid="stExpander"] {
        background-color: #ffffff;

        border: 1px solid #e2e8f0;

        border-radius: 10px;
    }

    [data-testid="stExpander"] summary {
        color: #1e293b !important;
    }


    /* =====================================================
       DIVIDERS
       ===================================================== */

    hr {
        border-color: #e2e8f0;
    }


    /* =====================================================
       REMOVE STREAMLIT BRANDING
       ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📦 MERIDIAN AI")

    st.caption("Supply Chain Intelligence")

    st.divider()

    st.subheader("⚙️ System")

    st.markdown("**🧠 AI Model**")
    st.caption("Llama 3.2 3B")

    st.markdown("**🔢 Embeddings**")
    st.caption("Nomic Embed Text")

    st.markdown("**🗄️ Vector Database**")
    st.caption("ChromaDB")

    st.markdown("**🔎 Retrieval**")
    st.caption("Top 6 chunks")

    st.divider()

    st.caption("Local AI • Ollama")
    st.caption("No OpenAI API required")


# =========================================================
# MAIN HEADER
# =========================================================

st.title("📦 Supply Chain Intelligence")

st.caption(
    "Internal knowledge assistant for procurement, "
    "supplier performance and supply-chain policies."
)


# =========================================================
# DASHBOARD METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="AI MODEL",
        value="Llama 3.2",
    )

with col2:
    st.metric(
        label="EMBEDDINGS",
        value="Nomic",
    )

with col3:
    st.metric(
        label="VECTOR DATABASE",
        value="ChromaDB",
    )

with col4:
    st.metric(
        label="RETRIEVAL",
        value="Top 6",
    )


st.write("")


# =========================================================
# DOCUMENT KNOWLEDGE BASE
# =========================================================

st.subheader("📄 Document Knowledge Base")

st.caption(
    "Upload procurement and supply-chain documents "
    "to make them searchable by the assistant."
)


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "Choose PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more supply-chain PDF documents.",
)


# =========================================================
# SELECTED FILES
# =========================================================

if uploaded_files:

    st.success(
        f"✓ {len(uploaded_files)} document(s) selected"
    )

    for uploaded_file in uploaded_files:

        file_col, size_col = st.columns(
            [5, 1]
        )

        with file_col:

            st.markdown(
                f"📄 **{uploaded_file.name}**"
            )

        with size_col:

            st.caption(
                f"{uploaded_file.size / 1024:.1f} KB"
            )


# =========================================================
# INDEX BUTTON
# Full-width separate row for proper alignment.
# =========================================================

st.write("")

index_button = st.button(
    "📚  Index Documents",
    type="primary",
    use_container_width=True,
)


# =========================================================
# INDEX DOCUMENTS
# =========================================================

if index_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF before indexing."
        )

    else:

        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        progress = st.progress(0)

        status = st.empty()

        try:

            # -------------------------------------------------
            # SAVE UPLOADED FILES
            # -------------------------------------------------

            status.info(
                "📄 Saving uploaded documents..."
            )

            for i, uploaded_file in enumerate(
                uploaded_files
            ):

                file_path = (
                    DATA_DIR / uploaded_file.name
                )

                with open(
                    file_path,
                    "wb",
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                progress.progress(
                    int(
                        ((i + 1) /
                         len(uploaded_files))
                        * 25
                    )
                )


            # -------------------------------------------------
            # LOAD DOCUMENTS
            # -------------------------------------------------

            status.info(
                "📖 Reading PDF documents..."
            )

            documents = load_documents()

            progress.progress(40)


            # -------------------------------------------------
            # SPLIT DOCUMENTS
            # -------------------------------------------------

            status.info(
                "✂️ Splitting documents into chunks..."
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                separators=[
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ],
            )

            chunks = splitter.split_documents(
                documents
            )

            progress.progress(55)


            # -------------------------------------------------
            # OLLAMA EMBEDDINGS
            # -------------------------------------------------

            status.info(
                "🧠 Creating Ollama embeddings..."
            )

            embeddings = OllamaEmbeddings(
                model="nomic-embed-text"
            )


            # -------------------------------------------------
            # CHROMADB
            # -------------------------------------------------

            status.info(
                "🗄️ Storing vectors in ChromaDB..."
            )

            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR,
            )

            vectorstore.add_documents(
                chunks
            )

            progress.progress(100)

            status.success(
                f"✓ Successfully indexed "
                f"{len(uploaded_files)} document(s) "
                f"and {len(chunks)} chunks."
            )

        except Exception as error:

            progress.empty()
            status.empty()

            st.error(
                f"❌ Indexing failed: {error}"
            )


# =========================================================
# KNOWLEDGE ASSISTANT
# =========================================================

st.divider()

st.subheader("💬 Knowledge Assistant")

st.caption(
    "Ask questions about the information contained "
    "in your indexed documents."
)


# =========================================================
# QUESTION
# =========================================================

question = st.text_area(
    "Your question",
    placeholder=(
        "Ask about suppliers, spend, delivery performance, "
        "procurement policy or sourcing requirements..."
    ),
    height=110,
)


# =========================================================
# ASK BUTTON
# =========================================================

ask_button = st.button(
    "🔎  Search & Ask",
    type="primary",
    use_container_width=True,
)


# =========================================================
# ANSWER
# =========================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "🔍 Searching the knowledge base..."
        ):

            try:

                # =================================================
                # RAG LOGIC — UNCHANGED
                # =================================================

                answer, sources = ask_question(
                    question,
                    top_k=6,
                )


                # =================================================
                # AI ANSWER
                # =================================================

                st.divider()

                st.subheader("🤖 AI Answer")

                st.write(answer)


                # =================================================
                # SOURCES
                # =================================================

                st.subheader("📚 Sources")

                if sources:

                    for source in sources:

                        with st.container(
                            border=True
                        ):

                            source_col, page_col = st.columns(
                                [5, 1]
                            )

                            with source_col:

                                st.markdown(
                                    f"📄 **{source['file']}**"
                                )

                            with page_col:

                                st.caption(
                                    f"Page {source['page']}"
                                )

                else:

                    st.info(
                        "No supporting sources were retrieved."
                    )


            except Exception as error:

                st.error(
                    f"❌ Unable to generate answer: {error}"
                )


# =========================================================
# EXAMPLE QUESTIONS
# =========================================================

st.divider()

with st.expander(
    "💡 Example Questions"
):

    example_col1, example_col2 = st.columns(2)

    with example_col1:

        st.markdown(
            """
### Supplier Performance

- Which supplier had the highest spend in Q1?
- What was the supplier's on-time delivery percentage?
- How many line stoppages happened in Q1?

### Procurement Policy

- What is the approval authority for a purchase
  order worth ₹1.4 crore?
- What are the four supplier classification categories?
- What qualifies a supplier as Critical?
"""
        )

    with example_col2:

        st.markdown(
            """
### Cross-Document Analysis

- Kaveri Metals recorded 88.1% on-time delivery
  and 1,150 defects per million. Which policy
  clauses does this trigger?

- The microcontroller supplier is single-source.
  What does the sourcing policy require?

- Microcontrollers are imported with a 46-day lead
  time. How many days of stock should be held?

### Trap Question

- What is the annual salary of the Head of Procurement?
"""
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Meridian Components Pvt. Ltd. • "
    "Supply Chain RAG Assistant • "
    "Powered by Ollama + ChromaDB"
)
