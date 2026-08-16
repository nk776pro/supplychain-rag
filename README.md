# 📦 Meridian Supply Chain Intelligence

A local AI-powered Supply Chain Knowledge Assistant built using Retrieval-Augmented Generation (RAG).

The application allows users to upload supply-chain PDF documents, index their contents, and ask natural-language questions about suppliers, procurement policies, delivery performance, sourcing, and other business information.

The project uses **Ollama** instead of the OpenAI API, allowing the language model to run locally.

---

## 🚀 Features

- 📄 Upload multiple PDF documents
- 📖 Extract text from PDF documents
- ✂️ Split documents into searchable chunks
- 🧠 Generate embeddings using Nomic Embed Text
- 🗄️ Store embeddings in ChromaDB
- 🔎 Retrieve relevant document sections
- 🤖 Generate answers using Llama 3.2 through Ollama
- 📚 Display supporting document sources and page numbers
- 💻 Run the entire AI pipeline locally
- 🔐 No OpenAI API key required
- 🛡️ Ground responses in uploaded documents to reduce hallucination

---

## 🏗️ System Architecture

```text
                         PDF Documents
                               │
                                                      ▼
                      Document Processing
                               │
                                                      ▼
                         Text Chunking
                               │
                                                      ▼
                     Nomic Embed Text
                        Embeddings
                               │
                                                      ▼
                           ChromaDB
                       Vector Database
                               │
                                                      ▼
                        User Question
                               │
                                                      ▼
                    Similarity Retrieval
                               │
                                                      ▼
                  Relevant Document Chunks
                               │
                                                      ▼
                    Llama 3.2 via Ollama
                               │
                                                      ▼
                          AI Answer
                               │
                                                      ▼
                    Supporting Sources
```

---

## 🛠️ Technologies Used

- **Python** — Application development
- **Streamlit** — Web interface
- **LangChain** — RAG pipeline and document processing
- **Ollama** — Local LLM execution
- **Llama 3.2** — Question answering
- **Nomic Embed Text** — Document embeddings
- **ChromaDB** — Vector database
- **PyPDF** — PDF text extraction

---

## 📂 Project Structure

```text
supplychain-rag/
├── app.py
├── rag.py
├── ingest.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── data/
```

### Main Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application and user interface |
| `rag.py` | Retrieval-Augmented Generation logic |
| `ingest.py` | PDF loading and document processing |
| `requirements.txt` | Python dependencies |
| `.env.example` | Example environment configuration |
| `README.md` | Project documentation |
| `data/` | PDF documents used by the application |

---

## 💻 Requirements

Before running the project, make sure you have:

- Ubuntu/Linux
- Python 3.10+
- pip
- Ollama
- Llama 3.2
- Nomic Embed Text

---

## ⚙️ Installation

### 1. Navigate to the project directory

```bash
cd ~/supplychain-rag
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

```bash
source venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the required Ollama models

```bash
ollama pull llama3.2
```

```bash
ollama pull nomic-embed-text
```

### 6. Verify the models

```bash
ollama list
```

---

## ▶️ Running the Application

Activate the virtual environment:

```bash
source venv/bin/activate
```

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

Open the address in your browser.

---

## 📄 How to Use

### Step 1 — Upload Documents

Open the application and use the **Document Knowledge Base** section.

Upload one or more supply-chain PDF documents.

### Step 2 — Index Documents

Click:

**📚 Index Documents**

The application will:

1. Save the uploaded documents.
2. Extract the text.
3. Split the text into chunks.
4. Generate embeddings using Nomic Embed Text.
5. Store the embeddings in ChromaDB.

### Step 3 — Ask Questions

Go to the **Knowledge Assistant** section.

Enter a question about the uploaded documents.

For example:

```text
Which supplier had the highest spend in Q1?
```

Click:

**🔎 Search & Ask**

### Step 4 — Review the Answer

The application displays:

- AI-generated answer
- Supporting documents
- Relevant page numbers

This allows users to verify the generated answer against the original documents.

---

## 🧪 Example Questions

### Supplier Performance

```text
Which supplier had the highest spend in Q1?
```

```text
What was the supplier's on-time delivery percentage?
```

```text
How many line stoppages happened in Q1?
```

### Procurement Policy

```text
What is the approval authority for a purchase order worth ₹1.4 crore?
```

```text
What are the four supplier classification categories?
```

```text
What qualifies a supplier as Critical?
```

### Cross-Document Analysis

```text
Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million. Which policy clauses does this trigger?
```

```text
The microcontroller supplier is single-source. What does the sourcing policy require?
```

```text
Microcontrollers are imported with a 46-day lead time. How many days of stock should be held?
```

---

## 🔎 Retrieval-Augmented Generation Pipeline

The project follows a Retrieval-Augmented Generation architecture.

### 1. Document Ingestion

PDF documents are uploaded and processed using the document ingestion pipeline.

### 2. Text Chunking

The extracted text is divided into smaller chunks so that relevant sections can be retrieved efficiently.

### 3. Embedding Generation

Each document chunk is converted into a numerical vector using **Nomic Embed Text**.

### 4. Vector Storage

The generated embeddings are stored in **ChromaDB**.

### 5. Question Processing

When a user asks a question, the system searches the vector database for the most relevant document chunks.

### 6. Context Retrieval

The most relevant chunks are retrieved and provided as context to the language model.

### 7. Local LLM Generation

Llama 3.2 running through Ollama generates the final answer using the retrieved context.

### 8. Source Display

The application displays supporting document names and page numbers along with the answer.

---

## 🤖 Why Ollama?

Instead of using the OpenAI API, this project uses Ollama to run the language model locally.

This provides several advantages:

- No OpenAI API key is required.
- No dependency on a cloud-based LLM API.
- Documents can remain on the local machine.
- The application can run using a local AI environment.
- The project can be demonstrated without an OpenAI API subscription.

---

## 🔐 Data and Privacy

The project is designed to operate locally.

PDF documents are processed by the local application, embeddings are stored in the local ChromaDB database, and the language model runs through Ollama on the local machine.

Users should avoid uploading confidential or sensitive documents unless the application environment is appropriately secured.

---

## 🛡️ Reducing Hallucination

The assistant is designed to answer questions using information retrieved from the uploaded documents.

If the required information is not available in the retrieved documents, the assistant can indicate that the information is not available rather than inventing a response.

This helps keep responses grounded in the uploaded knowledge base.

---

## ⚠️ Limitations

- The quality of answers depends on the quality and completeness of the uploaded documents.
- Documents must be indexed before they can be queried.
- The application currently works with PDF documents.
- Local LLM performance depends on available system resources.
- The assistant should not replace official procurement, financial, legal, or business decisions.
- If information is not present in the uploaded documents, the system may not be able to provide a reliable answer.

---

## 🔮 Future Improvements

Possible future improvements include:

- 💬 Conversation history
- 📚 Multiple knowledge bases
- 🗂️ Document management and deletion
- 🔎 Improved retrieval and reranking
- 📌 Better source highlighting
- 🔐 User authentication
- 👥 Document-level access permissions
- 📈 Analytics dashboard
- 🌐 Remote deployment
- 📱 Responsive interface
- ⚡ Faster document processing
- 🧠 Support for additional local LLMs

---

## 📦 Dependencies

The project uses the following Python packages:

```text
langchain
langchain-community
langchain-ollama
langchain-chroma
langchain-text-splitters
chromadb
pypdf
streamlit
python-dotenv
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🧹 Development Notes

The following directories are generated locally and do not need to be included in a source-code submission:

```text
venv/
__pycache__/
```

The ChromaDB directory can also be regenerated by indexing the documents through the application.

---

## 👨‍💻 Project Summary

**Meridian Supply Chain Intelligence** is a local Retrieval-Augmented Generation application designed to provide an AI-powered interface for querying supply-chain and procurement documents.

The system combines:

- Streamlit
- Python
- LangChain
- ChromaDB
- Nomic Embed Text
- Llama 3.2
- Ollama

to create a complete local document-question-answering pipeline.

The main goal is to allow users to interact with supply-chain documents using natural language while providing supporting sources for generated answers.

---

## ⭐ Project Highlights

```text
✓ Local AI
✓ Ollama-powered LLM
✓ PDF document processing
✓ RAG architecture
✓ Vector database
✓ Semantic retrieval
✓ Source-aware answers
✓ No OpenAI API dependency
✓ Streamlit user interface
```

---

**Meridian Supply Chain Intelligence**

*Local AI • Retrieval-Augmented Generation • Supply Chain Knowledge Assistant*
