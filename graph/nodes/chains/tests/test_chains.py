# How to run Pytest in the terminal
# In the terminal, enter "pytest . -s -v"
# "pytest ." - tells pytest to run from the current directory (the root directory of the project)
# "-s" - displays from stdout
# "-v" - is the verbose flag which shows the tests that we ran in pytest

# import pretty print
from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

from ..retrieval_grader import GradeDocuments, retrieval_grader_chain
from ..generation import generation_chain
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