import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_core.messages import HumanMessage
from pinecone import Pinecone

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# 1. Setup LLM (Llama 3 on Groq)
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 2. Setup Server-Side Embeddings (Saves RAM)
# Ensure your index was built with 'multilingual-e5-large' or update to match.
embeddings = PineconeEmbeddings(
    model="multilingual-e5-large", 
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# 3. Connect to Pinecone Vector Store
index_name = os.environ.get("PINECONE_INDEX", "adino-rag")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

def generate_answer(context_docs, question):
    context = "\n\n".join([d.page_content for d in context_docs])
    prompt = f"Answer ONLY from context: {context}\n\nQuestion: {question}"
    return llm.invoke([HumanMessage(content=prompt)]).content

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    role = data.get("role", "Employee")

    if not question: return jsonify({"error": "Question missing"}), 400

    # Retrieve docs and apply RBAC filter
    docs = vectorstore.similarity_search(question, k=6)
    allowed_docs = [
        d for d in docs 
        if role in d.metadata.get("allowed_roles", []) 
        or not d.metadata.get("allowed_roles")
    ]

    if not allowed_docs:
        return jsonify({"answer": "Access denied for your role."})

    return jsonify({"answer": generate_answer(allowed_docs, question)})

if __name__ == "__main__":
    # Bind to Render's dynamic port
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)