import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

DATA_PATH = "data/Updated_major_dataset.txt"

def get_allowed_roles(text: str):
    text = text.lower()
    roles = []

    if any(k in text for k in ["salary", "pf", "payroll", "joining", "induction", "bonus", "gratuity"]):
        roles.append("HR")

    if any(k in text for k in ["performance", "evaluation", "telecommuting", "conflict", "approval", "deputation"]):
        roles.append("Manager")

    if any(k in text for k in ["leave", "attendance", "dress code", "conduct", "holiday", "medical"]):
        roles.append("Employee")

    if any(k in text for k in ["welcome", "vision", "history", "mission", "company", "name", "about", "overview", "adino", "founded", "established"]):
        roles = ["HR", "Manager", "Employee"]

    if not roles:
        roles = ["HR", "Manager", "Employee"]

    return list(set(roles))

def build_pinecone_index():
    print("Loading handbook data...")
    loader = TextLoader(DATA_PATH, encoding="utf8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_text(docs[0].page_content)

    tagged_docs = []
    for chunk in chunks:
        roles = get_allowed_roles(chunk)
        tagged_docs.append(Document(page_content=chunk, metadata={"allowed_roles": roles}))

    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.environ.get("PINECONE_INDEX", "adino-rag")

    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=768,  # 768 for all-mpnet-base-v2
            metric="cosine",
        )

    print("Uploading documents to Pinecone...")
    PineconeVectorStore.from_documents(
        tagged_docs,
        embedding=embeddings,
        index_name=index_name,
    )
    print("Pinecone index built successfully.")

if __name__ == "__main__":
    build_pinecone_index()