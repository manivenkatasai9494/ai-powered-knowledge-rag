import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ---------------- LLM (Groq) ----------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ---------------- Pinecone ONLY (NO EMBEDDINGS LOADED) ----------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX", "adino-rag")

# IMPORTANT: embedding=None to avoid loading model in memory
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=None
)

def generate_answer(context_docs, question):
    context = "\n\n".join([d.page_content for d in context_docs])
    prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{question}
"""
    return llm.invoke([HumanMessage(content=prompt)]).content

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    role = data.get("role", "Employee")

    if not question:
        return jsonify({"error": "Question missing"}), 400

    docs = vectorstore.similarity_search(question, k=5)

    allowed_docs = [
        d for d in docs
        if role in d.metadata.get("allowed_roles", [])
        or not d.metadata.get("allowed_roles")
    ]

    if not allowed_docs:
        return jsonify({"answer": "You do not have access to this information."})

    answer = generate_answer(allowed_docs, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
