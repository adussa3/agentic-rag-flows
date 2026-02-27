# We will use this (question) router to implement a version of Adaptive RAG
# Adapative RAG is basically using a question router to route our question to different RAG flows
#
# We're going to use two RAG flows:
# (1) Taking the route to WEB_SEARCH and search the internet, and continue downstream with the same logic/flow we had before
# (2) Using the retrieval augmentation from the vector store
#
# We're going to take the user's question and determine if it's stroed in the vector store to answer that question.
# If it's not, we'll take the route to web search and answer from there

# Literal makes it so that a variable of this typing can only take one predefined set of value
# This is useful for validation and type checking
from typing import Literal

# The ChatPromptTemplate is a class that holds our content that we send to the LLM as a human message,
# or that we receive back from the LLM as an answer that's tagged as an assistant message
from langchain_core.prompts import ChatPromptTemplate

# We use Pydantic's BaseModel and Field to create the object schemas to structure the output of the LLM's response
# NOTE: We're prompting the LLM through the Field description!
from pydantic import BaseModel, Field

# We're importing RunnableSequence for type hinting
from langchain_core.runnables import RunnableSequence

# Import LangChain OpenAI
from langchain_openai import ChatOpenAI

# Create a structured output class
class RouteQuery(BaseModel):
  """Route a user query to the most relevant datasource"""
  datasource: Literal["vectorstore", "websearch"] = Field(
    # The ellipsis (...) means that this field is required for any object instantiations of this class
    ...,
    description="Given a user question choose to route it to web search or vectorstore."
  )

# Initialize Structured LLM
# The response we get back from the LLM is no an AIMessage, but the RouteQuery Pydantic class
llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(RouteQuery)

# The System prompt
SYSTEM_PROMPT = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on thise topics. For all else, use web-search."""

# Create a Chat Prompt Template with .from_messages()
route_prompt = ChatPromptTemplate.from_messages([
  ("system", SYSTEM_PROMPT),
  ("human", "{question}")
])

question_router_chain: RunnableSequence = route_prompt | structured_llm_grader