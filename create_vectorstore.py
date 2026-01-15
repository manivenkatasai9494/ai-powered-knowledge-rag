import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def get_allowed_roles(text):
    text = text.lower()
    roles = []

    if any(k in text for k in ["salary", "pf", "bonus", "payroll", "gratuity"]):
        roles.append("HR")
    if any(k in text for k in ["approval", "performance", "manager", "deputation"]):
        roles.append("Manager")
    if any(k in text for k in ["leave", "attendance", "policy", "dress"]):
        roles.append("Employee")

    return roles if roles else ["HR", "Manager", "Employee"]

def build_pinecone_index():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "Updated_major_dataset.txt")

    loader = TextLoader(DATA_PATH, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = splitter.split_text(docs[0].page_content)

    documents = [
        Document(
            page_content=c,
            metadata={"allowed_roles": get_allowed_roles(c)}
        )
        for c in chunks
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX", "adino-rag")

    existing = [i.name for i in pc.list_indexes()]
    if index_name in existing:
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=384,     # IMPORTANT (MiniLM = 384)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    PineconeVectorStore.from_documents(
        documents,
        embedding=embeddings,
        index_name=index_name
    )

    print("✅ Pinecone index created successfully")

if __name__ == "__main__":
    build_pinecone_index()
