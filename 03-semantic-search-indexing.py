import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

## 读取 PDF
# CHROMA_LOCATION = os.getenv("CHROMA_LOCATION")
file_path = "E:/workspace/my_rag/pdfs/2025Shimano.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

# print (len(docs))
# print (type(docs[0]))
# print (docs[0])

## 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, ## by char
    chunk_overlap = 200,
    add_start_index = True
)

all_splits = text_splitter.split_documents(docs)
print (len(all_splits))
print ("1st", all_splits[0].page_content)
# print ("2nd", all_splits[1].page_content)
# print ("3rd", all_splits[2].page_content)

## 向量化
embedding = OllamaEmbeddings(
    model="qwen3-embedding:4b"
)

# vector_0 = embedding.embed_query(all_splits[0].page_content)
# print (vector_0)
# print (len(vector_0))

vector_store =  Chroma(
    collection_name = "example_collection",
    embedding_function = embedding,
    persist_directory = "./chroma_langchain_db"
)
ids = vector_store.add_documents(documents=all_splits)

print (len(ids))