import os, bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import shutil

if os.path.exists("./chroma_rag_db"):
    shutil.rmtree("./chroma_rag_db")

page_url = "https://starwalk.space/zh-Hans/news/3i-atlas-interstellar-object"

bs4_strainer = bs4.SoupStrainer()

loader = WebBaseLoader(
    web_path=(page_url),
    bs_kwargs={"parse_only": bs4_strainer}
)

docs = loader.load()

## 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200, ## by char
    chunk_overlap = 50,
    add_start_index = True
)

all_splits = text_splitter.split_documents(docs)
for i in range(len(all_splits)):
    print (f"💡 {i}")
    print (all_splits[i].page_content)

## 向量化
embedding = OllamaEmbeddings(
    model="qwen3-embedding:4b"
)
vector_store =  Chroma(
    collection_name = "rag_collection",
    embedding_function = embedding,
    persist_directory = "./chroma_rag_db"
)
ids = vector_store.add_documents(documents=all_splits)