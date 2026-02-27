# How to run Pytest in the terminal
# In the terminal, enter "pytest . -s -v"
# "pytest ." - tells pytest to run from the current directory (the root directory of the project)
# "-s" - displays from stdout
# "-v" - is the verbose flag which shows the tests that we ran in pytest

# import pretty print
from pprint import pprint

from ..retrieval_grader import GradeDocuments, retrieval_grader_chain
from ..generation import generation_chain
from ..hallucination_grader import GradeHallucinations, hallucination_grader_chain
from ..answer_grader import GradeAnswer, answer_grader_chain
from ..router import RouteQuery, question_router_chain
from ingestion import retriever

def test_retrieval_grader_answer_yes() -> None:
  question = "agent memory"
  docs = retriever.invoke(question)
  doc_text = docs[0].page_content

  response: GradeDocuments = retrieval_grader_chain.invoke({"document": doc_text, "question": question})
  assert response.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
  question = "agent memory"
  docs = retriever.invoke(question)
  doc_text = docs[0].page_content
  
  grade_document: GradeDocuments = retrieval_grader_chain.invoke({"document": doc_text, "question": "how to make pizza"})
  assert grade_document.binary_score == "no"

# NOTE: This won't be an actual test, it's just to make sure everything is working as expected
def test_generation_chain() -> None:
  question = "agent memory"

  # Get relevant documents
  docs = retriever.invoke(question)

  # Run the generation chain with the retrieved documents as the context
  response = generation_chain.invoke({"context": docs, "question": question})
  pprint(response)

def test_hallucination_grader_answer_yes() -> None:
  question = "agent memory"
  docs = retriever.invoke(question);
  generation = generation_chain.invoke({"context": docs, "question": question});

  respone: GradeHallucinations = hallucination_grader_chain.invoke({"document": docs, "generation": generation})
  assert respone.binary_score

def test_hallucination_grader_answer_no() -> None:
  question = "agent memory"
  docs = retriever.invoke(question);
  generation = "blah, blah, blah"

  response: GradeHallucinations = hallucination_grader_chain.invoke({"document": docs, "generation": generation})
  print(response)
  assert not response.binary_score

def test_answer_grader_yes() -> None:
  question = "agent memory"
  docs = retriever.invoke(question)
  generation = generation_chain.invoke({"context": docs, "question": question})

  response: GradeAnswer = answer_grader_chain.invoke({"question": question, "generation": generation})
  assert response.binary_score

def test_answer_grader_no() -> None:
  question = "agent memory"
  generation = "In order to make pizza, you first need to start with the dough"

  response: GradeAnswer = answer_grader_chain.invoke({"question": question, "generation": generation})
  assert response.binary_score

def test_route_to_vectorstore() -> None:
  question = "agent memory"
  response: RouteQuery = question_router_chain.invoke({"question": question})
  assert response.datasource == "vectorstore"

def test_route_to_websearch() -> None:
  question = "how to make pizza"
  response: RouteQuery = question_router_chain.invoke({"question": "how to make pizza"})
  assert response.datasource == "websearch"