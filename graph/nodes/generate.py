# Any and Dict will be used for Type Hinting
from typing import Any, Dict

# Import Generation Chain
from chains.generation import generation_chain

# import GraphState
from graph.state import GraphState

# Create Generate node
def generation_node(state: GraphState) -> Dict[str, Any]:
  print("---GENERATE---")
  question = state["question"]
  docs = state["documents"]
  response = generation_chain.invoke({"context": docs, "question": question})
  return {"generation": response}