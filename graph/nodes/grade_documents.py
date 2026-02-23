# Any and Dict will be used for Type Hinting
from typing import Any, Dict

from chains.retrieval_grader import retrieval_grader_chain

# GraphState is the input for our node and what it'll update
from graph.state import GraphState

# The state will hold the retrieved documents
# The Grade Documents will iterate through the documents and determine if they are relevent to the user's question
def grade_documents_node(state: GraphState) -> Dict[str, Any]:
  """
  Determines whether the retrieved documents are relevant to the question
  If any document is not relevant, we will set a flag to run web search

  Args:
    state (dict): The current graph state
  
  Returns:
    state (dict): Filtered out irrelevant documents and updated web_search state
  """
  print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
  question = state["question"]
  documents = state["documents"]

  filtered_docs = []
  web_search = False

  for doc in documents:
    score = retrieval_grader_chain.invoke({"document": doc.page_content, "question": question})
    grade = score.binary_score
    if grade.lower() == "yes":
      print("---GRADE: DOCUMENT RELEVANT---")
      filtered_docs.append(doc)
    else:
      print("---GRADE: DOCUMENT NOT RELEVANT---")
      web_search = True
  
  return {"document": filtered_docs, "web_search": web_search}