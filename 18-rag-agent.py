from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool

load_dotenv()

## RAG
embedding = OllamaEmbeddings(model="qwen3-embedding:4b")
vector_store = Chroma(
    collection_name = "rag_collection",
    embedding_function = embedding,
    persist_directory = "./chroma_rag_db"
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve information to help answer a query
    
    :param query: User's query
    :type query: str
    """
    retrieved_docs = vector_store.similarity_search(query, k=2)
    content = '\n\n'.join(
        (f"Source: {doc.metadata}\nContent:{doc.page_content}") for doc in retrieved_docs
    )
    return content, retrieved_docs

system_prompt = """
    你是一个负责任的、诚实的好孩子，如果你不知道某些知识，就不应该
    胡编乱造，你应该诚实地说不知道；否则，你就会死。
    
    你的回答必须简洁，因为你一个高冷男神，不会说废话。

    今天是2025年12月28日。
"""

## Agent 定义
local_model = init_chat_model(
    # model = "ollama:deepseek-r1:1.5b",
    model = "ollama:llama3.2:3b",
    base_url = "http://localhost:11434",
    temperature = 0.5,
    max_tokens = 2000
)
agent = create_agent(
    model="deepseek:deepseek-chat",
    # model = local_model,
    tools = [retrieve_context],
    system_prompt=system_prompt
)

## as-is Chat Result
results = agent.invoke(
    {"messages": [{"role": "user", "content": "3i/atlas 什么时候会撞击地球？"}]}
)
for i in results["messages"]:
    i.pretty_print()