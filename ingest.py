from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "supply_chain"


def load_documents():
    documents = []

    for pdf_file in DATA_DIR.glob("*.pdf"):

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_file.name

        documents.extend(pages)

    return documents


def ingest_documents():

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vectorstore.add_documents(chunks)

    return len(documents), len(chunks)


if __name__ == "__main__":

    pages, chunks = ingest_documents()

    print(f"Loaded {pages} pages.")
    print(f"Stored {chunks} chunks.")
    print("Ingestion complete!")
