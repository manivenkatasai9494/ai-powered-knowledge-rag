import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

def build_pinecone_index():
    loader = TextLoader("Updated_major_dataset.txt", encoding="utf8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_text(docs[0].page_content)

    # Use server-side embeddings
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    index_name = os.environ.get("PINECONE_INDEX", "adino-rag")

    PineconeVectorStore.from_texts(
        chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print("Index updated successfully.")

if __name__ == "__main__":
    build_pinecone_index()