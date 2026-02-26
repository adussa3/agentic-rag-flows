# The answergrader_chain will grade the answer and determine whether 
# the LLM's generated response actually answers the question or not

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
class GradeAnswer(BaseModel):
  """"""
  binary_score: bool = Field("Answer addressed the question, 'yes' or 'no'")

# Initialize Structured LLM
# The response we get back from the LLM is not an AIMessage, but the GradeAnswer Pydantic class
llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# The System prompt
SYSTEM_PROMPT = """You are a grader assesing whether an answer addresses / resolved a question \n
          Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

# Create a Chat Prompt Template with .from_messages()
answer_prompt = ChatPromptTemplate.from_messages([
  ("system", SYSTEM_PROMPT),
  ("human", "User question: \n\n {question} \n\n LLM generation: \n\n {generation}")
])

# The answer_grader_chain's response will be a GradeAnswer Pydantic model
answer_grader_chain: RunnableSequence = answer_prompt | structured_llm_grader