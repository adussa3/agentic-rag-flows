from typing import List, TypedDict

class GraphState(TypedDict):
  """
  Represents the state of our graph.
  
  Attributes:
    question: question - user's question
    generation: LLM generation - the generated answer to the user's question
    web_search: whether to add search - boolean flag to tell us if we need to search online for extra results
    documents: list of documents - documents to help us answer the user's question
  """
  question: str
  generation: str
  web_search: bool
  documents: List[str]
