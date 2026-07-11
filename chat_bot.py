import os

from flask import Flask, render_template, request, jsonify     
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from rag_system import Rag_Core

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

app = Flask(__name__)

rag_core = Rag_Core()
embeddings = rag_core.create_embeddings()
vector_store = rag_core.load_vector_store(embeddings)


def build_answer(query, vector_store, chat_history=None):
    chat_history = chat_history or []

    retrieval_query = query
    if chat_history:
        previous_question = chat_history[-1][0]
        retrieval_query = f"{query} (related to previous question: {previous_question})"

    docs = vector_store.similarity_search(retrieval_query, k=3)
    if not docs:
        return "I could not find relevant information in the document."

    context = "\n\n".join(doc.page_content for doc in docs)

    history_text = ""
    if chat_history:
        recent_history = chat_history[-4:]
        history_text = "\n".join(
            f"User: {user_q}\nAssistant: {assistant_a}"
            for user_q, assistant_a in recent_history
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question using the supplied context. If the question is a follow-up, use the conversation history as well. If the context is insufficient, say so clearly.",
            ),
            (
                "human",
                "Conversation history:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"history": history_text, "context": context, "question": query})
    return response.content


def chatting(vector_store):
    chat_history = []
    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            print("\nSession ended.")
            break

        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue
        
        answer = build_answer(query, vector_store, chat_history)
        chat_history.append((query, answer))
        print("Bot:", answer, "\n")


# if __name__ == "__main__":
#     rag_core = Rag_Core()
#     embeddings = rag_core.create_embeddings()
#     #vector_store = rag_core.store_vectors_in_pinecone(embeddings)
#     vector_store = rag_core.load_vector_store(embeddings)
#     chatting(vector_store)
        
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=['POST'])
def chat():
    print("chat")
    data = request.get_json(silent=True) or {}
    user_input = data.get("user_input") or data.get("message") or ""
    print(f"Received user input: {user_input}")

    if not user_input:
        return jsonify({"error": "No user input provided."}), 400

    try:
        answer = build_answer(user_input, vector_store)
    except Exception as exc:
        return jsonify({"error": str(exc), "response": "I could not process your request right now."}), 500

    return jsonify({"answer": answer, "response": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)