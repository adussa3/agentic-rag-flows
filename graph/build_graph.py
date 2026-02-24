import sys
import os

# ------------------------------
# Add project root to sys.path
# ------------------------------
# Ensures we can do `from graph.*` and `from langgraph.*` from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


###########
# IMPORTS #
###########

# load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Import END node and StateGraph
from langgraph.graph import END, StateGraph

# Import node name consts
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH

# Import nodes
from graph.nodes import generate_node, grade_documents_node, retrieve_node, web_search_node

# Import our custom GraphState
from graph.state import GraphState

##############################
# CONDITIONAL EDGE FUNCTIONS #
##############################

def decide_to_generate(state: GraphState):
  print("---ASSESS GRADED DOCUMENTS---")
  web_search = state["web_search"]

  if web_search:
    # We found a document that's not relevant to the user's question
    print("---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION---")
    return WEB_SEARCH
  
  print("---DECISION: GENERATE---")
  return GENERATE

#########
# GRAPH #
#########

builder = StateGraph(state_schema=GraphState)

# Nodes
builder.add_node(RETRIEVE, retrieve_node)
builder.add_node(GRADE_DOCUMENTS, grade_documents_node)
builder.add_node(WEB_SEARCH, web_search_node)
builder.add_node(GENERATE, generate_node)

# Entry Point
builder.set_entry_point(RETRIEVE)

# Edges
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)
builder.add_edge(WEB_SEARCH, GENERATE)
builder.add_edge(GENERATE, END)

# Conditional Edges
builder.add_conditional_edges(
  GRADE_DOCUMENTS,
  decide_to_generate,
  {
    WEB_SEARCH: WEB_SEARCH,
    GENERATE: GENERATE
  }
)

# Compile graph
graph = builder.compile()

###################
# VISUALIZE GRAPH #
###################

# This prints the graph visualization in the console using mermaid syntax
# We can paste this mermaid code in the mermaid live editor (https://mermaid.live/) to see the graph visualization
print(graph.get_graph().draw_mermaid())

# This prints the graph visualization in the console using ascii characters
# Note: you need to install Gandalf to view the graph visualization
print(graph.get_graph().draw_ascii())

# This saves the graph visualization as a png file
with open("RAG.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())