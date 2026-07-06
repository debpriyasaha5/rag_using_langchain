from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from rag_system import Rag_Core

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


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


if __name__ == "__main__":
    rag_core = Rag_Core()
    embeddings = rag_core.create_embeddings()
    vector_store = rag_core.store_vectors_in_pinecone(embeddings)
    chatting(vector_store)
        