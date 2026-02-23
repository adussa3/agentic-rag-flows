# We're going to download the prompt from the hub
# from langchain import hub (DEPRECATED)
from langsmith import Client

# The StrOutputParser takes in our message, get's it's content, and returns it into a string
from langchain_core.output_parsers import StrOutputParser

# AI Model
from langchain_openai import ChatOpenAI

# Initialize AI model
llm = ChatOpenAI(temperature=0)

# Pull RAG prompt from the LangChain hub
# This is a very standard RAG prompt that the LangChain team wrote
# It gives the LLM the role of an assistant for question answering, plugging in the context
# Which is going to be all the retrieved documents, web search result, and the original question
#
# """
# You are an assistant for question-answering tasks.
# Use the following pieces of retrieved context to answer the question.
# If you don't know the answer, just say that you don't know.
# Use three sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Answer:
# """
#
# https://smith.langchain.com/hub/rlm/rag-prompt
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")

# Generation Chain
# We pipe the prompt into the llm, then pipe the response into StrOutputParser to return a string (instead of AIMessage)
generation_chain = prompt | llm | StrOutputParser()