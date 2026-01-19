from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding = OllamaEmbeddings(model = "qwen3-embedding:4b")

score_measures = [
    "default",
    "cosine",   # 向量夹角
    "l2",       # 向量距离
    "ip"        # 向量内积/点积
]

# 创建向量库和4个coll
persist_dir = "./chroma_score_db"
vector_stores = []
for score_measure in score_measures:
    collection_metadata = {"hnsw:space": score_measure}
    if score_measure == "default":
        collection_metadata = None
    
    collection_name = f"my_collection_{score_measure}"
    vector_stores.append(Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_dir,
        collection_metadata=collection_metadata
    ))

def indexing(docs):
    print ("\n加入文档：")
    for vector_store in vector_stores:
        ids = vector_store.add_documents(docs)
        print (f"\n集合：{vector_store._collection.name}")
        print (ids)

def query_with_score(query):
    for i in range(len(score_measures)):
        results = vector_stores[i].similarity_search_with_score(query, k=2)
        print (f"\n💡搜索：{query}")
        for doc, score in results:
            print (doc.page_content, end='')
            print (f"\n✅ {score_measures[i]}: {score}")



# file_path = "E:/workspace/my_rag/pdfs/智能体白皮书.pdf"
# loader = PyPDFLoader(file_path)
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 200, ## by char
#     chunk_overlap = 50,
#     add_start_index = True
# )
# all_splits = text_splitter.split_documents(docs)
# # print (all_splits[10].page_content)
# indexing(all_splits)

query_with_score("思维链是用来干嘛的？")