# vector DB + retrieval
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def build_vector_store():

    docs = [
        "Credit utilization above 30% is risky.",
        "High debt-to-income ratios increase default risk.",
        "Emergency savings should cover 3-6 months of expenses."
    ]

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(docs, embeddings)

    return db