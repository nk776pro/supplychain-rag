from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "supply_chain"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.1,
)


PROMPT = """
You are an internal supply chain assistant for Meridian Components Pvt. Ltd.

Answer the user's question ONLY using the context provided below.

Rules:
1. Do not use outside knowledge.
2. Do not make up numbers, rules, names, or policies.
3. If the answer is not available in the provided context, say:
   "The information is not available in the uploaded documents."
4. If the question requires information from multiple documents, combine
   the relevant information from the context.
5. Give a clear and concise answer.
6. When useful, mention the relevant policy clause or document.

Context:
{context}

Question:
{question}
"""


def ask_question(question, top_k=6):
    results = vectorstore.similarity_search(
        question,
        k=top_k
    )

    if not results:
        return (
            "The information is not available in the uploaded documents.",
            []
        )

    context_parts = []

    for doc in results:
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template(PROMPT)

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    sources = []

    for doc in results:
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page", 0) + 1

        item = {
            "file": source,
            "page": page
        }

        if item not in sources:
            sources.append(item)

    return response.content, sources
