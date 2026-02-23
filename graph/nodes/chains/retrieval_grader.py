# The ChatPromptTemplate is a class that holds our content that we send to the LLM as a human message,
# or that we receive back from the LLM as an answer that's tagged as an assistant message
from langchain_core.prompts import ChatPromptTemplate

# We use Pydantic's BaseModel and Field to create the object schemas to structure the output of the LLM's response
# NOTE: We're prompting the LLM through the Field description!
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Create Pydantic model
class GradeDocuments(BaseModel):
  """Binary score for relevance check on retrieved documents"""
  binary_score: str = Field("Documents are relevant to the question, 'yes' or 'no'")

# Initialize LLM
# The LLM needs to support with_structured_output/function calling
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# with_structured_output converts the LLM's response to the GradeDocuments Pydantic model
structured_llm_grader = llm.with_structured_output(GradeDocuments)

SYSTEM_PROMPT = """You are a grader accessing relevance of a retrieved document to a user question. \n
  If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
  Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
"""

#
grade_prompt = ChatPromptTemplate.from_messages(
  [
    ("system", SYSTEM_PROMPT),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
  ]
)

# The retrieval grader chain uses the LLM's output and turn it into a Pydantic object 
# which has the information whether the document is relevant or not
#
# It filters out documents that are NOT relevant and only keeps the relevant documents
#
# Additionally, if at least one document is not relevent to the user's question,
# it'll mark the state's web search flag to true
retrieval_grader_chain = grade_prompt | structured_llm_grader