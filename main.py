from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# import Graph
from graph.build_graph import graph

if __name__ == '__main__':
  print("Hello Advanced RAG")
  print(graph.invoke(input={"question": "what is agent memory?"}))
