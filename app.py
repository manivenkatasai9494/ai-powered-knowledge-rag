import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",   # or any Groq model you like
    groq_api_key=os.getenv("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# ---------- Pinecone vector store ----------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ.get("PINECONE_INDEX", "adino-rag")

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

def generate_answer(context_docs, question):
    context = "\n\n".join([d.page_content for d in context_docs])

    prompt = f"""
You are the Adino Knowledge Assistant.
Answer ONLY from the context below.
If not allowed, say you do not have access.

Context:
{context}

Question:
{question}

Answer:
"""
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

    # 1️⃣ Retrieve from Pinecone
    docs = vectorstore.similarity_search(question, k=6)

    # 2️⃣ RBAC filter
    allowed_docs = [
        d for d in docs
        if role in d.metadata.get("allowed_roles", [])
        or not d.metadata.get("allowed_roles")  # fallback
    ]

    if not allowed_docs:
        return jsonify({
            "answer": "I do not have access to that information for your current role."
        })

    answer = generate_answer(allowed_docs, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)