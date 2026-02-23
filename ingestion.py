# Before we implement a RAG agent, we first need to index our documents into a vector store
# (1) We're going to load articles into LangChain documents
# (2) Chunk the articles into smaller pieces
# (3) Embed the chunks
# (4) Store the embeddings into ChromaDB (an open source vector store)

# DISCLAIMER: we're implementing a very simple ingestion pipeline
# A true ingestion process will have a lot of optimizations in the pipeline
# 
# In this project, we're focusing on the retrieval part, NOT the ingestion

# load environment variables from .env file
from dotenv import load_dotenv

# Split up the documents
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the documents from the internet
from langchain_community.document_loaders import WebBaseLoader

# Chroma Vector Store
from langchain_chroma import Chroma

# OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

# Load the URLs into LangChain documents
docs = [WebBaseLoader(url).load() for url in urls]

# Flatten the docs list
docs_list = [item for sublist in docs for item in sublist]

# Create text spliter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=250, chunk_overlap=0
)

# Chunk documents
doc_chunks = text_splitter.split_documents(docs_list)

# Index chunks in Chroma DB
# NOTE: we only need to run this once!
#       We don't want to index this everytime we run the program
#       We simply want to load everything from the disk
#
# vector_store = Chroma.from_documents(
#   documents=doc_chunks,
#   collection_name="rag-chroma",
#   # Embeds the chunks - OpenAIEmbeddings defaults to small-embeddings-3
#   embedding=OpenAIEmbeddings(),
#   # Persist vector store into our disk
#   persist_directory="./.chroma"
# )

# Create LangChain Retriever object from the ChromaDB
# This performs similarity searches
retriever = Chroma(
  collection_name="rag-chroma",
  persist_directory="./.chroma",
  embedding_function=OpenAIEmbeddings()
).as_retriever()