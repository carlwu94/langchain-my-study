import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 4. 核心：构建与Ollama本地模型对话的链
from langchain.chains.retrieval import create_retrieval_chain
from langchain import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama  # 关键：导入Ollama

loader = PyPDFLoader("E:\workspace\my_rag\pdfs\三花智控：2022年三季度报告.pdf")  # 请替换为你的文件路径
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
local_model_path = r"./models/sentence-transformers_all-MiniLM-L6-v2" 
embeddings = HuggingFaceEmbeddings(model_name=local_model_path)

db = FAISS.from_documents(texts, embeddings)