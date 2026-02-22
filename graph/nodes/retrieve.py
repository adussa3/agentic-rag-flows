# Any and Dict will be used for Type Hinting
from typing import Any, Dict

# GraphState is the input for our node and what it'll update
from graph.state import GraphState

# The retriever references our local ChromaDB vector store with the stored embeddings
from ingestion import retriever


# The Retrieve node extracts the user's question and retrieves the relevant documents from the graph state
# The node returns a Dict with want to update in the state
def retrieve_node(state: GraphState) -> Dict[str, Any]:
  print("---RETRIEVE---")

  # Get the user's question from the state
  question = state["question"]

  # The retrieve performs a sementic search and retrieves the relevant documents
  documents = retriever.invoke(question)

  # We want to update teh field of documents in our current state with the retrieved documents
  return {"documents": documents}