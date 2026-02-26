# the hallucination_grader chain is going to determien whether the gnerated answer we get from the LLM
# is grounded in the documents

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
class GradeHallucinations(BaseModel):
  """Binary score for hallucination present in generation answer."""
  binary_score: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no")

# Initialize Structured LLM
# The response we get back from the LLM is no an AIMessage, but the GradeHallucinations Pydantic class
llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# The System prompt
SYSTEM_PROMPT = """You are a grader accessing whether an LLM genration is grounded in / supported by a supported by a set of documents.
          Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# Create a Chat Prompt Template with .from_messages()
hallucination_prompt = ChatPromptTemplate.from_messages([
  ("system", SYSTEM_PROMPT),
  ("human", "Set of facts: \n\n {document} \n\n LLM generation: {generation}")
])

# The hallucination_grader_chain's response will be a GradeHallucinations Pydantic model
hallucination_grader_chain: RunnableSequence = hallucination_prompt | structured_llm_grader