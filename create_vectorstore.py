import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

DATA_PATH = "data/Updated_major_dataset.txt"
INDEX_PATH = "vectorstore"

def get_allowed_roles(text):
    text = text.lower()
    roles = []
    # [cite_start]HR Role: Personnel, Payroll, and Benefits [cite: 135, 451, 457]
    if any(k in text for k in ["salary", "pf", "payroll", "joining", "induction", "bonus", "gratuity"]):
        roles.append("HR")
    # [cite_start]Manager Role: Performance and Approvals [cite: 365, 380, 418]
    if any(k in text for k in ["performance", "evaluation", "telecommuting", "conflict", "approval", "deputation"]):
        roles.append("Manager")
    # [cite_start]Employee Role: General Conduct and Leave [cite: 256, 468, 480]
    if any(k in text for k in ["leave", "attendance", "dress code", "conduct", "holiday", "medical"]):
        roles.append("Employee")
    # [cite_start]Public: Common Company Info [cite: 1, 26, 47]
    # Expanded to include company name, about, overview, etc.
    if any(k in text for k in ["welcome", "vision", "history", "mission", "company", "name", "about", "overview", "adino", "founded", "established"]):
        roles = ["HR", "Manager", "Employee"]
    
    # If no roles found, make it accessible to all (for general information)
    if not roles:
        roles = ["HR", "Manager", "Employee"]
    
    return list(set(roles))

def build_index():
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
    vectorstore = FAISS.from_documents(tagged_docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print("Vectorstore created with HR, Manager, and Employee roles.")

if __name__ == "__main__":
    build_index()