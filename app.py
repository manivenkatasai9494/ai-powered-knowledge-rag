import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_core.messages import HumanMessage

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# 1. Initialize LLM (Llama 3 via Groq) [cite: 1]
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 2. Server-side Embeddings (Critical for Render memory limits) [cite: 1, 2]
embeddings = HuggingFaceEmbeddings(model_naimport os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.messages import HumanMessage
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# 1. Initialize LLM (Llama 3 via Groq)
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 2. API-based Embeddings (Saves RAM for 512MB limit)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACE_API_KEY"), 
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# 3. Connect to Pinecone Index
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ.get("PINECONE_INDEX", "adino-rag")

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

def generate_answer(context_docs, question):
    context = "\n\n".join([d.page_content for d in context_docs])
    prompt = f"Answer ONLY from the context provided.\nContext: {context}\n\nQuestion: {question}"
    return llm.invoke([HumanMessage(content=prompt)]).content

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    role = data.get("role", "Employee")

    if not question:
        return jsonify({"error": "Question missing"}), 400

    # Retrieve and RBAC filter
    docs = vectorstore.similarity_search(question, k=5)
    allowed_docs = [
        d for d in docs
        if role in d.metadata.get("allowed_roles", [])
        or not d.metadata.get("allowed_roles")
    ]

    if not allowed_docs:
        return jsonify({"answer": "I do not have access to that information for your role."})

    answer = generate_answer(allowed_docs, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)me="all-mpnet-base-v2")

# 3. Connect to Pinecone Index [cite: 1, 2]
index_name = os.environ.get("PINECONE_INDEX", "adino-rag")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

def generate_answer(context_docs, question):
    context = "\n\n".join([d.page_content for d in context_docs])
    prompt = f"Answer ONLY from the context provided.\nContext: {context}\n\nQuestion: {question}"
    return llm.invoke([HumanMessage(content=prompt)]).content

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    role = data.get("role", "Employee")

    if not question:
        return jsonify({"error": "Question missing"}), 400

    # Retrieve and RBAC filter [cite: 1, 2]
    docs = vectorstore.similarity_search(question, k=5)
    allowed_docs = [
        d for d in docs
        if role in d.metadata.get("allowed_roles", [])
        or not d.metadata.get("allowed_roles")
    ]

    if not allowed_docs:
        return jsonify({"answer": "Access denied for your role."})

    answer = generate_answer(allowed_docs, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    # Fix Port Binding for Render [cite: 1]
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)