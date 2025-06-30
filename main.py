from typing import List
from utils.vector_store import VectorStore
from utils.agent import create_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PDF_PATH = "data/Stock_Market_Performance_2024.pdf"

FALLBACK_MESSAGE = (
    "I don't have information for this. "
    "You can ask about stock market performance in 2024."
)


def main():
    # Initialize FAISS vector store and agent
    vector_store = VectorStore()
    agent = create_agent()

    # Extract text from PDF and load into FAISS vector store
    pdf_text = vector_store.extract_text_from_pdf(PDF_PATH)
    vector_store.create_vector_store([pdf_text])

    print("RAG Chatbot (LangGraph + FAISS) is ready! Type your question (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        # Search for relevant context
        results = vector_store.similarity_search(user_input, k=2)
        if not results or all(r.page_content.strip() == '' for r in results):
            print(f"Bot: {FALLBACK_MESSAGE}")
            continue
        # Prepare context for agent
        context = "\n".join([r.page_content for r in results])
        prompt = f"Context: {context}\n\nQuestion: {user_input}"
        response = agent.invoke({"input": prompt, "messages": []})
        print(f"Bot: {response.content}")

if __name__ == "__main__":
    main()
