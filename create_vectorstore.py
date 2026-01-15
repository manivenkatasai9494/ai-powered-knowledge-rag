import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

load_dotenv()

def build_pinecone_index():
    # Use the path to your data folder 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "Updated_major_dataset.txt")

    loader = TextLoader(DATA_PATH, encoding="utf8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_text(docs[0].page_content)

    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    index_name = os.environ.get("PINECONE_INDEX", "adino-rag")

    PineconeVectorStore.from_texts(
        chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print("Pinecone index is ready!")

if __name__ == "__main__":
    build_pinecone_index()