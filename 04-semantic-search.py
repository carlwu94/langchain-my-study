from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embedding = OllamaEmbeddings(
    model="qwen3-embedding:4b"
)
vector_store = Chroma(
    collection_name = "example_collection",
    embedding_function = embedding,
    persist_directory = "./chroma_langchain_db"
)

# 1.相似度查询
# results = vector_store.similarity_search(
#     "三花智控的最大持股股东是谁？"
# )
# for index, result in enumerate(results):
#     print ("💡", index)
#     print (result.page_content)


# 2.带分数的相似度查询
# results = vector_store.similarity_search_with_score(
#     "Vanquish纺车轮有哪些强大的功能？"
# )
# for (doc, score) in results:  # unpacking tuple
#     print ("💡", score)
#     print (doc.page_content)


# 3.用向量进行相似度查询
# vector = embedding.embed_query(
#     "万奎士卖多少钱"
# )
# results = vector_store.similarity_search_by_vector(vector)
# for index, result in enumerate(results):
#     print ("💡", index)
#     print (result.page_content)


# 4.用修饰器进行查询
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=2)

results = retriever.invoke("Shimano纺车轮有哪些当家的技术？")
for index, result in enumerate(results):
    print ("💡", index)
    print (result.page_content)