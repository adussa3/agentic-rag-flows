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

# Import hallucination_grader_chain, answer_grader_chain, and question_router_chain used in a conditional edge
from graph.nodes.chains.hallucination_grader import GradeHallucinations, hallucination_grader_chain
from graph.nodes.chains.answer_grader import GradeAnswer, answer_grader_chain
from graph.nodes.chains.router import RouteQuery, question_router_chain

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

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
  print("---CHECK HALLUCINATIONS---")
  question = state["question"]
  docs = state["documents"]
  generation = state["generation"]

  hallucination_score: GradeHallucinations = hallucination_grader_chain.invoke({"document": docs, "generation": generation})
  if hallucination_grade := hallucination_score.binary_score:
    print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
  else:
    print("---DECISION: GENERATION IS NOT GROUDED IN DOCUMENTS---")
    return "not supported"

  print("---GRADE GENERATION vs QUESTION---")
  answer_score: GradeAnswer = answer_grader_chain.invoke({"question": question, "generation": generation})
  if answer_grade := answer_score.binary_score:
    print("--DECISION: GENERATION ADDRESSES QUESTION")
    return "useful"
  else:
    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
    return "not useful"
  
def route_question(state: GraphState) -> str:
  question = state["question"]
  source: RouteQuery = question_router_chain.invoke({"question": question})
     
  if source.datasource == "websearch":
    print("---ROUTE QUESTION TO WEB SEARCH---")
    return WEB_SEARCH

  print("---ROUTE QUESTION TO VECTOR STORE---")
  return RETRIEVE


#########
# GRAPH #
#########

builder = StateGraph(state_schema=GraphState)

# Nodes
builder.add_node(RETRIEVE, retrieve_node)
builder.add_node(GRADE_DOCUMENTS, grade_documents_node)
builder.add_node(WEB_SEARCH, web_search_node)
builder.add_node(GENERATE, generate_node)

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

builder.add_conditional_edges(
  GENERATE,
  grade_generation_grounded_in_documents_and_question,
  # The key strings is what will be displayed for the edges
  {
    "not supported": GENERATE,
    "useful": END,
    "not useful": WEB_SEARCH 
  }
)

# Entry Point
# builder.set_entry_point(RETRIEVE)

# Conditional Entry Point - a conditional edge with the first node as the entry point
builder.set_conditional_entry_point(
  route_question,
  {
    WEB_SEARCH: WEB_SEARCH,
    RETRIEVE: RETRIEVE
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
# print(graph.get_graph().draw_ascii())

# This saves the graph visualization as a png file
with open("Adaptive RAG.png", "wb") as f:
  f.write(graph.get_graph().draw_mermaid_png())