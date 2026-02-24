# Any and Dict will be used for Type Hinting
from typing import Any, Dict

# import LangChain Document class
from langchain_core.documents import Document

# The TavilySearch class is a LangChain tool that runs the Tavily search engine on queries we provide it
from langchain_tavily import TavilySearch

# import GraphState
from graph.state import GraphState

# load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Tavily Search Results Tool
# max_results==3 guarantees we would get at most 3 results
web_search_tool = TavilySearch(max_results=3)

# NOTE: We execute the websearch node only AFTER we filter the documents with the grade_documents node
# Therefore, we're not suppose to have any non-relevant documents
def web_search_node(state: GraphState) -> Dict[str, Any]:
  print("---WEB SEARCH---")
  question = state["question"]
  documents = state["documents"]

  # We want to take the content of the search results and combine them into a LangChain Document
  tavily_results = web_search_tool.invoke({"query": question})["results"]

  # We join the content together to create one big string
  joined_tavily_results = "\n".join(
    [result["content"] for result in tavily_results]
  )

  # We create a LangChain Document with our joined string
  web_results = Document(page_content=joined_tavily_results)

  # Add web_results to documents
  if documents is not None:
    documents.append(web_results)
  else:
    documents = [web_results]

  return {"documents": documents}

if __name__ == "__main__":
  web_search_node(state={"question": "agent memory", "documents": None})