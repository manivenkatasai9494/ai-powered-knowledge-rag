import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def get_allowed_roles(text: str):
    text = text.lower()
    roles = []
    if any(k in text for k in ["salary", "pf", "payroll", "joining", "induction", "bonus", "gratuity"]):
        roles.append("HR")
    if any(k in text for k in ["performance", "evaluation", "telecommuting", "conflict", "approval", "deputation"]):
        roles.append("Manager")
    if any(k in text for k in ["leave", "attendance", "dress code", "conduct", "holiday", "medical"]):
        roles.append("Employee")
    if any(k in text for k in ["welcome", "vision", "history", "mission", "company", "adino"]):
        roles = ["HR", "Manager", "Employee"]
    return list(set(roles)) if roles else ["HR", "Manager", "Employee"]

def build_pinecone_index():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "Updated_major_dataset.txt")

    loader = TextLoader(DATA_PATH, encoding="utf8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_text(docs[0].page_content)

    tagged_docs = [Document(page_content=c, metadata={"allowed_roles": get_allowed_roles(c)}) for c in chunks]

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    index_name = os.environ.get("PINECONE_INDEX", "adino-rag")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=768, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    PineconeVectorStore.from_documents(tagged_docs, embedding=embeddings, index_name=index_name)
    print("Index ready!")

if __name__ == "__main__":
    build_pinecone_index()