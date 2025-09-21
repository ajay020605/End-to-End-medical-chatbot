from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from functools import lru_cache

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
def translate_to_english(text, llm):
    translation_prompt = f"Translate the following text into English,Do not explain, just return translation and if it is already in english return the same text:\n\n{text}"
    result = llm.invoke(translation_prompt)
    return result.content

# Prompt template
prompt_template = (
    "You are a helpful and empathetic medical assistant.\n\n"
    "Here is some reference information from our knowledge base:\n"
    "{context}\n\n"
    "User's question: {original_input}\n\n"
    "Chat history: {chat_history}\n\n"
    "If the context contains relevant medical info, use it naturally in your reply.\n"
    "If the context does not help, respond politely that the information is not available in the knowledge base.\n"
    "Use the chat history to maintain continuity and remember what has already been discussed and based on that give the user reply according to the context .\n"
    "You may also ask the user gentle follow-up questions if you need more details about their situation in order to give a better and safer response.\n"
    "Always reply in the same language as the user's original question ({original_input}).\n"
    "Always sound natural, supportive, and strictly used context and don't make it sound like you have use the context for generating response,reply naturally."
)
prompt = ChatPromptTemplate.from_template(prompt_template)


@lru_cache(maxsize=1)
def get_rag_chain():
    """Load all components lazily, cache after first run"""
    embeddings = download_hugging_face_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medicalbot",
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        max_output_tokens=500,
        google_api_key=GOOGLE_API_KEY,
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "msg" not in data:
            return jsonify({"error": "Invalid request"}), 400

        user_message = data["msg"]
        chat_history = data.get("chat_history", [])  # default empty list if not provided
        rag_chain = get_rag_chain()  # Lazy load only once
        # Translate user_message → English for retrieval
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            max_output_tokens=200,
            google_api_key=GOOGLE_API_KEY,
        )
        translated_query = translate_to_english(user_message, llm)
        print("Translated query:", translated_query)
        response = rag_chain.invoke({
            "input": translated_query,
            "original_input": user_message,
            "chat_history": chat_history
        })
        return jsonify({"answer": response["answer"]})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/health")
def health():
    return "OK", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)