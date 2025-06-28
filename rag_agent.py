import os
from typing import List, Dict, Any, TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from PyPDF2 import PdfReader
from gpt4all import GPT4All

@dataclass
class Document:
    page_content: str
    metadata: dict = None

    def __post_init__(self):
        self.metadata = self.metadata or {}

# Configuration
PDF_PATH = "Stock_Market_Performance_2024.pdf"
MODEL_NAME = "orca-mini-3b-gguf2-q4_0.ggml"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Define the state
class GraphState(TypedDict):
    query: str
    documents: List[Document]
    context: str
    response: str
    llm: Any = None

def load_llm():
    """Load the GPT4All model"""
    try:
        print("Loading GPT4All model (this may take a while on first run)...")
        # List available models
        print("Available models:", GPT4All.list_models())
        
        # Try to find a working model
        working_models = [
            "orca-mini-3b-gguf2-q4_0.ggml",
            "gpt4all-falcon-q4_0.gguf",
            "ggml-model-gpt4all-falcon-q4_0.bin"
        ]
        
        for model in working_models:
            try:
                print(f"\nTrying model: {model}")
                llm = GPT4All(model)
                print(f"Successfully loaded model: {model}")
                return llm
            except Exception as e:
                print(f"Failed to load {model}: {e}")
        
        print("\nNo working model found. Trying to download a default model...")
        return GPT4All("orca-mini-3b-gguf2-q4_0.ggml")
        
    except Exception as e:
        print(f"Critical error loading LLM: {e}")
        return None

def load_and_split_documents() -> List[Document]:
    """Load and split the PDF document into chunks."""
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found at {PDF_PATH}")
    
    print("Loading PDF document...")
    documents = []
    try:
        reader = PdfReader(PDF_PATH)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Simple text splitter
                words = text.split()
                for i in range(0, len(words), CHUNK_SIZE):
                    chunk = ' '.join(words[i:i+CHUNK_SIZE])
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"page": page_num + 1, "source": PDF_PATH}
                    ))
    except Exception as e:
        print(f"Error reading PDF: {e}")
        raise
    
    if not documents:
        raise ValueError(f"No readable text found in {PDF_PATH}")
    
    return documents

class SimpleRetriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
    
    def get_relevant_documents(self, query: str, k: int = 3) -> List[Document]:
        query_terms = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            content = doc.page_content.lower()
            score = sum(1 for term in query_terms if term in content)
            if score > 0:  # Only include documents with at least one matching term
                scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]

def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieve relevant documents for the query."""
    retriever = state["retriever"]
    state["documents"] = retriever.get_relevant_documents(state["query"])
    state["context"] = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page {doc.metadata.get('page', 'N/A')}:\n{doc.page_content}"
        for doc in state["documents"]
    )
    return state

def generate_response(state: GraphState) -> GraphState:
    """Generate a response using GPT4All."""
    context = state.get("context", "")
    query = state["query"]
    llm = state.get("llm")
    
    if not llm:
        state["response"] = "LLM not loaded. Please check the model installation."
        return state
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Please provide a clear and concise answer based on the context above. If the context doesn't contain the answer, say "I don't have enough information to answer this question."

Answer:"""
    
    try:
        response = llm.generate(prompt, max_tokens=500, temp=0.7)
        # Handle different response formats
        if hasattr(response, 'choices') and response.choices:
            state["response"] = response.choices[0].text.strip()
        else:
            state["response"] = str(response)
    except Exception as e:
        state["response"] = f"Error generating response: {str(e)}"
    
    return state

class RAGAgent:
    def __init__(self):
        # Initialize attributes first
        self.documents = None
        self.retriever = None
        self.llm = None
        self.workflow = None
        
        # Load components
        try:
            print("Loading documents...")
            self.documents = load_and_split_documents()
            self.retriever = SimpleRetriever(self.documents)
            print("Loading language model...")
            self.llm = load_llm()
            
            if not self.llm:
                raise ValueError("Failed to load language model")
                
            # Create the workflow
            self.workflow = self._create_workflow()
            
        except Exception as e:
            print(f"Error initializing RAGAgent: {str(e)}")
            raise

    def _create_workflow(self):
        """Create and compile the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Add nodes with proper binding
        workflow.add_node("retrieve", lambda state: retrieve_documents(state))
        workflow.add_node("generate", lambda state: generate_response(state))
        
        # Define edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Set the entry point
        workflow.set_entry_point("retrieve")
        
        # Compile the workflow
        return workflow.compile(checkpointer=None)  # Disable checkpointer for simplicity

    def query(self, question: str) -> str:
        """Query the RAG agent."""
        if not self.retriever:
            return "Error: Document retriever not initialized."
        if not self.llm:
            return "Error: Language model not loaded."
            
        try:
            # Initialize state with all required fields
            state = {
                "query": question,
                "documents": [],
                "context": "",
                "response": "",
                "retriever": self.retriever,
                "llm": self.llm
            }
            
            # Run the workflow
            result = self.workflow.invoke(state)
            return result.get("response", "No response generated")
            
        except Exception as e:
            return f"Error processing query: {str(e)}\n{type(e).__name__}: {str(e)}"

def main():
    """Example usage of the RAG agent with LangGraph and GPT4All."""
    try:
        print("Initializing RAG agent with GPT4All...")
        agent = RAGAgent()
        
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ").strip()
            if not question:
                continue
            if question.lower() == 'quit':
                break
            
            print("\nProcessing your question...")
            try:
                response = agent.query(question)
                print("\nResponse:", response)
            except Exception as e:
                print(f"\nError: {str(e)}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()